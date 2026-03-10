#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/extension/extension_glsl_compiler.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/utils/profiler.hpp"
#include <iostream>
#include <random>

#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

int main() {
    spdlog::set_level(spdlog::level::debug);

    std::vector<std::string> context_extensions = {"merian-resources", "merian-glsl-compiler"};
#ifndef NDEBUG
    context_extensions.push_back("merian-validation-layers");
#endif

    // Setup Vulkan context.
    const merian::ContextCreateInfo context_create_info{
        .features = merian::VulkanFeatures({"hostQueryReset", "pushDescriptor"}),
        .context_extensions = context_extensions,
        .application_name = "merian-example",
    };
    const merian::ContextHandle context = merian::Context::create(context_create_info);
    const auto ext_res = context->get_context_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = ext_res->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    std::vector<float> array(1024 * 1024 * 512, 1.f);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0., 1.);

    {
        MERIAN_PROFILE_SCOPE(profiler, fmt::format("generate {} random numbers", array.size()));
        for (std::size_t i = 0; i < array.size(); i++) {
            array[i] = dist(rng);
        }
    }

    {
        MERIAN_PROFILE_SCOPE(profiler, "CPU compute");
        double sum = 0.0; // need to use doubles else the result is off by a lot
        for (uint32_t i = 0; i < array.size(); i++) {
            sum += array[i];
        }
        std::cout << "CPU sum = " << sum << std::endl;
    }
    {
        merian::CommandPoolHandle cmd_pool = merian::CommandPool::create(queue);
        merian::CommandBufferHandle cmd = merian::CommandBuffer::create(cmd_pool);
        cmd->begin();

        MERIAN_PROFILE_SCOPE(profiler, "GPU compute");

        const uint32_t workgroup_size_x = 32;
        const uint32_t workgroup_size_y = 32;
        const uint32_t workgroup_size = workgroup_size_x * workgroup_size_y;

        merian::BufferHandle buffer1, buffer2;
        {
            MERIAN_PROFILE_SCOPE(profiler, "allocate and upload");

            buffer1 = alloc->create_buffer(array.size() * sizeof(float),
                                           vk::BufferUsageFlagBits::eStorageBuffer,
                                           merian::MemoryMappingType::HOST_ACCESS_RANDOM);
            buffer2 = alloc->create_buffer(sizeof(float) * (array.size() / workgroup_size + 1),
                                           vk::BufferUsageFlagBits::eStorageBuffer,
                                           merian::MemoryMappingType::HOST_ACCESS_RANDOM);

            void* buffer1_mapped = buffer1->get_memory()->map();
            memcpy(buffer1_mapped, array.data(), array.size() * sizeof(float));
            buffer1->get_memory()->unmap();
        }

        merian::GLSLShaderCompilerHandle compiler =
            context->get_context_extension<merian::ExtensionGLSLCompiler>()->get_compiler();
        merian::PipelineHandle pipe;
        {
            MERIAN_PROFILE_SCOPE(profiler, "create pipeline");

            const auto desc_layout = merian::DescriptorSetLayoutBuilder()
                                         .add_binding_storage_buffer()
                                         .add_binding_storage_buffer()
                                         .build_push_descriptor_layout(context);
            const auto shader = compiler->find_compile_glsl_to_entry_point(
                context, "src/compute_sum.comp", context->get_shader_compile_context());
            const auto pipe_layout = merian::PipelineLayoutBuilder(context)
                                         .add_descriptor_set_layout(desc_layout)
                                         .add_push_constant<uint32_t>()
                                         .build_pipeline_layout();
            merian::SpecializationInfoBuilder spec_builder;
            vk::PhysicalDeviceSubgroupProperties subgroup_props =
                context->get_physical_device()->get_properties();
            spec_builder.add_entry(workgroup_size_x, workgroup_size_y, subgroup_props.subgroupSize);
            const merian::SpecializationInfoHandle spec_info = spec_builder.build();

            pipe = merian::ComputePipeline::create(pipe_layout, shader, spec_info);
        }

        std::array<merian::BufferHandle, 2> ping_pong_buffers = {buffer1, buffer2};
        uint32_t ping_pong_i = 0;
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "record commands");
            uint32_t current_size = array.size();

            uint32_t iteration = 0;
            while (current_size > 1) {
                MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("iteration {}", iteration++));
                cmd->bind(pipe);
                cmd->push_descriptor_set(pipe, ping_pong_buffers[ping_pong_i],
                                         ping_pong_buffers[ping_pong_i ^ 1]);
                cmd->push_constant(pipe, current_size);

                const uint32_t group_count_x =
                    (uint32_t)ceil(sqrt((current_size + workgroup_size - 1) / workgroup_size));
                const uint32_t group_count_y =
                    (uint32_t)ceil(sqrt((current_size + workgroup_size - 1) / workgroup_size));
                cmd->dispatch(group_count_x, group_count_y, 1);

                const auto bar1 = ping_pong_buffers[ping_pong_i]->buffer_barrier(
                    vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite);
                const auto bar2 = ping_pong_buffers[ping_pong_i ^ 1]->buffer_barrier(
                    vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader, {bar1, bar2});

                ping_pong_i ^= 1;
                current_size /= workgroup_size;
            }
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "submit and wait");

            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eHost,
                         ping_pong_buffers[ping_pong_i]->buffer_barrier(
                             vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead));
            cmd->end();
            queue->submit_wait(cmd);
        }

        float result;
        {
            MERIAN_PROFILE_SCOPE(profiler, "download and print result");
            result = *ping_pong_buffers[ping_pong_i]->get_memory()->map_as<float>();
            std::cout << "GPU sum = " << result << std::endl;
            ping_pong_buffers[ping_pong_i]->get_memory()->unmap();
        }
    }

    profiler->collect();
    std::cout << merian::Profiler::get_report_str(profiler->get_report()) << std::endl;

    return 0;
}
