cmake_minimum_required(VERSION 3.18)
project(seedvr2_ncnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- ncnn (Vulkan build) install ----
set(NCNN_DIR "F:/TencentOCR/ncnn/build-ncnn-gpu/install" CACHE PATH "ncnn install dir")
set(VULKAN_SDK "C:/VulkanSDK/1.4.350.0" CACHE PATH "Vulkan SDK")

include_directories(${NCNN_DIR}/include)
include_directories(${VULKAN_SDK}/Include)

set(NCNN_LIBS
    ${NCNN_DIR}/lib/libncnn.a
    ${NCNN_DIR}/lib/libglslang.a
    ${NCNN_DIR}/lib/libglslang-default-resource-limits.a
    ${NCNN_DIR}/lib/libOSDependent.a
    ${NCNN_DIR}/lib/libSPIRV.a
    ${NCNN_DIR}/lib/libMachineIndependent.a
    ${NCNN_DIR}/lib/libGenericCodeGen.a
    ${VULKAN_SDK}/Lib/vulkan-1.lib
)

find_package(OpenMP REQUIRED)

function(add_seedvr2_exe name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
    target_compile_options(${name} PRIVATE -fopenmp -O2)
    target_link_options(${name} PRIVATE
        -static-libgcc -static-libstdc++
        -Wl,--allow-multiple-definition
    )
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
endfunction()

# M0 烟雾测试（旧 main.cpp 已被端到端 seedvr2_run 替换）
# add_seedvr2_exe(seedvr2_test src/main.cpp)
# M1 验证：patchify+linear(vid_in) + fusedrms RMSNorm
add_seedvr2_exe(seedvr2_verify_m1 src/verify_m1.cpp)
# M2 验证：单 block attn(qkv/qk_norm/out)+mlp(SwiGLU)，block 0 与 block 20
add_seedvr2_exe(seedvr2_verify_m2 src/verify_m2.cpp)
# M3 验证：AWA 算子（window partition + mmrope3d + varlen concat + 窗口内 SDPA + reverse）
add_seedvr2_exe(seedvr2_verify_awa src/verify_awa.cpp)
# M4 验证：shifted AWA（变长窗口 + 文本跨窗口平均 coalesce）
add_seedvr2_exe(seedvr2_verify_awa_shifted src/verify_awa_shifted.cpp)
# M5 调试：block 0 逐子层分诊
add_seedvr2_exe(seedvr2_verify_dit_dbg0 src/verify_dit_dbg0.cpp)

# M5 验证：完整 32 层 NaDiT 单次前向（C++ 忠实复现）对拍 ref_vid_out.bin
add_seedvr2_exe(seedvr2_verify_dit src/verify_dit.cpp)

# M6 (Vulkan) — SeedVR2 NaDiT 推理引擎（GEMM 走 ncnn Vulkan Net）
add_library(seedvr2_dit STATIC src/dit_vk.cpp src/awa_vk.cpp src/awa_layer.cpp src/ada_compose.cpp src/cast_bf16_layer.cpp src/dit_graph.cpp src/seedvr2_pipeline.cpp)
target_include_directories(seedvr2_dit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${NCNN_DIR}/include ${VULKAN_SDK}/Include)
target_link_libraries(seedvr2_dit ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_dit PRIVATE -fopenmp -O2)

# ---------- AWA 自定义 Vulkan shader 编译（glslangValidator -> models/m5/*.spv） ----------
find_program(GLSLANG_VALIDATOR "glslangValidator"
    PATHS "C:/VulkanSDK/1.4.350.0/Bin" "C:/VulkanSDK/1.3.0.0/Bin")
if(GLSLANG_VALIDATOR)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa.spv
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/awa.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/awa.comp
        COMMENT "Compile awa.comp -> models/m5/awa.spv"
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa_coalesce.spv
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/awa_coalesce.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa_coalesce.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/awa_coalesce.comp
        COMMENT "Compile awa_coalesce.comp -> models/m5/awa_coalesce.spv"
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa_init.spv
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/awa_init.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa_init.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/awa_init.comp
        COMMENT "Compile awa_init.comp -> models/m5/awa_init.spv"
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose.spv
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/ada_compose.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/ada_compose.comp
        COMMENT "Compile ada_compose.comp -> models/m5/ada_compose.spv"
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose_bf16.spv
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/ada_compose_bf16.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose_bf16.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/ada_compose_bf16.comp
        COMMENT "Compile ada_compose_bf16.comp -> models/m5/ada_compose_bf16.spv"
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose_fp16.spv
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/ada_compose_fp16.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose_fp16.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/ada_compose_fp16.comp
        COMMENT "Compile ada_compose_fp16.comp -> models/m5/ada_compose_fp16.spv"
    )
    add_custom_target(awa_spv ALL DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa.spv
        ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa_coalesce.spv
        ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/awa_init.spv
        ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose.spv
        ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose_bf16.spv
        ${CMAKE_CURRENT_SOURCE_DIR}/models/m5/ada_compose_fp16.spv)
else()
    message(WARNING "glslangValidator 未找到；需手动把 awa.spv / awa_coalesce.spv / awa_init.spv 放到 models/m5/")
endif()
# 验证：Vulkan 前向数值对拍 ref_vid_out.bin
add_executable(seedvr2_dit_vk_test src/dit_vk_test.cpp)
target_link_libraries(seedvr2_dit_vk_test seedvr2_dit)
# 阶段1 单测：AwaLayer（自定义 ncnn::Layer）vs AwaVk 参考 cos=1.0
add_executable(seedvr2_test_awa_layer src/test_awa_layer.cpp)
target_link_libraries(seedvr2_test_awa_layer seedvr2_dit)
# 阶段0 探针：VkMat 连续执行链路（input(VkMat)->extract(VkMat,cmd) 全程 GPU）
add_executable(seedvr2_probe_vkmat_io src/probe_vkmat_io.cpp)
target_link_libraries(seedvr2_probe_vkmat_io seedvr2_dit)
# 阶段2 验证：整图 dit_graph vs forward_latent cos=1.0
add_executable(seedvr2_quick_binop src/quick_binop.cpp)
target_link_libraries(seedvr2_quick_binop seedvr2_dit)
add_executable(seedvr2_min_awa_load src/min_awa_load.cpp)
target_link_libraries(seedvr2_min_awa_load seedvr2_dit)
add_executable(seedvr2_test_dit_graph src/test_dit_graph.cpp src/awa_window.cpp)
target_link_libraries(seedvr2_test_dit_graph seedvr2_dit)
# 单 Net（32 层合并 1 个 Net）vs 分块图 数值对比
add_executable(seedvr2_test_single_net src/test_single_net.cpp src/awa_window.cpp)
target_link_libraries(seedvr2_test_single_net seedvr2_dit)
# 端到端 exe（文件 IO：读 vid_grid/txt/windows → 写 sr_latent）
add_executable(seedvr2_dit_vk_run src/dit_vk_run.cpp)
target_link_libraries(seedvr2_dit_vk_run seedvr2_dit)

# M8 验证：自定义 Vulkan AWA 数值对拍（GPU 路径 vs CPU 参考）
add_executable(seedvr2_verify_awa_vk src/verify_awa_vk.cpp)
target_link_libraries(seedvr2_verify_awa_vk seedvr2_dit)

# M8 单元自测：极小合成输入，秒级复现/隔离 AwaVk 崩溃
add_executable(seedvr2_test_awa_unit src/test_awa_unit.cpp)
target_link_libraries(seedvr2_test_awa_unit seedvr2_dit)

# M8 隔离测试：最小 shader 验证 plumbing（12 绑定 + push constant + download）
add_executable(seedvr2_test_awa_min src/test_awa_min.cpp)
target_link_libraries(seedvr2_test_awa_min ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_test_awa_min PRIVATE -fopenmp -O2)
target_link_options(seedvr2_test_awa_min PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# Vulkan 上传/绑定/回读链路诊断：copy shader 原样拷回比对
add_executable(seedvr2_test_copy src/test_copy.cpp)
target_link_libraries(seedvr2_test_copy ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_test_copy PRIVATE -O2)
target_link_options(seedvr2_test_copy PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# Vulkan GEMM 最小探针（隔离 GPU 崩溃）
add_executable(seedvr2_probe_vk src/probe_vk.cpp)
target_link_libraries(seedvr2_probe_vk ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_probe_vk PRIVATE -fopenmp -O2)
target_link_options(seedvr2_probe_vk PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# VAE 子图 ncnn Vulkan 前向验证（vae_enc1/enc2/dec1/dec2）
add_executable(seedvr2_verify_vae_vk src/verify_vae_vk.cpp)
target_link_libraries(seedvr2_verify_vae_vk ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_verify_vae_vk PRIVATE -fopenmp -O2)
target_link_options(seedvr2_verify_vae_vk PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# attention 子图中间层 dump（定位哪层出错）
add_executable(seedvr2_dump_attn_layers src/dump_attn_layers.cpp)
target_link_libraries(seedvr2_dump_attn_layers ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_dump_attn_layers PRIVATE -fopenmp -O2)
target_link_options(seedvr2_dump_attn_layers PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# 最小 MHA 测试（2D 输入）
add_executable(seedvr2_verify_mha_min src/verify_mha_min.cpp)
target_link_libraries(seedvr2_verify_mha_min ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_verify_mha_min PRIVATE -fopenmp -O2)
target_link_options(seedvr2_verify_mha_min PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# VAE 完整 encode/decode 验证
add_executable(seedvr2_verify_vae_full src/verify_vae_full.cpp src/vae_vk.cpp)
target_link_libraries(seedvr2_verify_vae_full ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_verify_vae_full PRIVATE -fopenmp -O2)
target_link_options(seedvr2_verify_vae_full PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# 窗口生成验证
add_executable(seedvr2_verify_window src/verify_window.cpp src/awa_window.cpp)
target_link_libraries(seedvr2_verify_window ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_verify_window PRIVATE -fopenmp -O2)
target_link_options(seedvr2_verify_window PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# 全 Vulkan 自包含端到端 CLI（读图/帧序列 -> VAE encode -> DiT -> VAE decode -> LAB -> 写图/逐帧）
add_executable(seedvr2_run src/main.cpp src/vae_vk.cpp src/awa_window.cpp src/image_io.cpp src/color_fix.cpp src/seedvr2_engine.cpp)
target_include_directories(seedvr2_run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(seedvr2_run seedvr2_dit ${NCNN_LIBS} OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_run PRIVATE -fopenmp -O2 -ffp-contract=off)
target_link_options(seedvr2_run PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# LAB 校正验证
add_executable(seedvr2_verify_colorfix src/verify_colorfix.cpp src/color_fix.cpp)
target_link_libraries(seedvr2_verify_colorfix OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_verify_colorfix PRIVATE -fopenmp -O2)
target_link_options(seedvr2_verify_colorfix PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)

# RNG 验证（rng::Randn 逐位对齐 PyTorch CPU torch.randn）
add_executable(seedvr2_verify_rng src/verify_rng.cpp)
target_link_libraries(seedvr2_verify_rng OpenMP::OpenMP_CXX -pthread)
target_compile_options(seedvr2_verify_rng PRIVATE -fopenmp -O2 -ffp-contract=off)
target_link_options(seedvr2_verify_rng PRIVATE -static-libgcc -static-libstdc++ -Wl,--allow-multiple-definition)
