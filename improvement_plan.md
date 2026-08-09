 For Phoenix/780M, I would pursue these in order:                                                                                                                                                               
                                                                                                                                                                                                                
 ### 1. Tune prefill chunk size                                                                                                                                                                                 
                                                                                                                                                                                                                
 The current default is 1024. Test:                                                                                                                                                                             
                                                                                                                                                                                                                
 ```sh                                                                                                                                                                                                          
   for n in 256 512 1024 1536 2048; do                                                                                                                                                                          
     ./q36-bench --vulkan --prefill-chunk $n \                                                                                                                                                                  
       --prompt-file tests/long_context_story_prompt.txt \                                                                                                                                                      
       --ctx-start 2048 --ctx-max 2048 --gen-tokens 8                                                                                                                                                           
   done                                                                                                                                                                                                         
 ```                                                                                                                                                                                                            
                                                                                                                                                                                                                
 Larger chunks may improve GEMM reuse and reduce host orchestration, while smaller chunks may improve occupancy and memory pressure.                                                                            
                                                                                                                                                                                                                
 ### 2. Investigate cooperative matrix support                                                                                                                                                                  
                                                                                                                                                                                                                
 The captured Vulkan profile reports (but the static capture is incomplete):                                                                                                                                                                           
                                                                                                                                                                                                                
 ```text                                                                                                                                                                                                        
   VK_KHR_cooperative_matrix                                                                                                                                                                                    
 ```                                                                                                                                                                                                            
                                                                                                                                                                                                                
 This could provide a substantial gain for:                                                                                                                                                                     
                                                                                                                                                                                                                
 - moe_gate_up_gemm                                                                                                                                                                                             
 - moe_down_gemm                                                                                                                                                                                                
 - dense Q8 GEMMs                                                                                                                                                                                               
                                                                                                                                                                                                                
 First query supported matrix shapes/types at runtime with
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`; the static
`VP_VULKANINFO` JSON does not enumerate them. If FP16 cooperative matrices are
returned on RADV Phoenix, prototype only one GEMM kernel before redesigning the
whole backend.                                    
                                                                                                                                                                                                                
 ### 3. Test integer dot products for dense Q8                                                                                                                                                                  
                                                                                                                                                                                                                
 The hardware exposes Vulkan integer-dot-product support. Current dense kernels perform many scalar unpack/dequant/f16 FMAs. A Q8 path using packed int8 dot products could reduce instruction count            
 significantly.                                                                                                                                                                                                 
                                                                                                                                                                                                                
 Target:                                                                                                                                                                                                        
                                                                                                                                                                                                                
 ```text                                                                                                                                                                                                        
   vulkan/matmul_q8_0_mm_f16.comp                                                                                                                                                                               
   vulkan/matmul_q8_0_mm_f16_out32.comp                                                                                                                                                                         
 ```                                                                                                                                                                                                            
                                                                                                                                                                                                                
 This is especially promising because dense Q8 GEMMs consume approximately 27% of profiled time.                                                                                                                
                                                                                                                                                                                                                
 ### 4. Reconsider the Q2/IQ2 format for prefill                                                                                                                                                                
                                                                                                                                                                                                                
 IQ2 saves memory bandwidth but has expensive irregular dequantization. Test a comparable Q4 model or more regular Q2/Q4 expert format.                                                                         
                                                                                                                                                                                                                
 On this GPU, a larger but more regular Q4 kernel may outperform IQ2 during prompt processing because:                                                                                                          
                                                                                                                                                                                                                
 - shared DDR5 bandwidth is limited;                                                                                                                                                                            
 - IQ2 unpacking is instruction-heavy;                                                                                                                                                                          
 - current MoE GEMMs already dominate prefill.                                                                                                                                                                  
                                                                                                                                                                                                                
 Do not assume fewer bits means higher prompt throughput.                                                                                                                                                       
                                                                                                                                                                                                                
 ### 5. Phoenix-specific MoE tiling                                                                                                                                                                             
                                                                                                                                                                                                                
 The current GEMMs use:                                                                                                                                                                                         
                                                                                                                                                                                                                
 ```text                                                                                                                                                                                                        
   256 threads                                                                                                                                                                                                  
   64 output rows                                                                                                                                                                                               
   32 token slots                                                                                                                                                                                               
 ```                                                                                                                                                                                                            
                                                                                                                                                                                                                
 Benchmark variants with:                                                                                                                                                                                       
                                                                                                                                                                                                                
 - 32 rows × 32 slots                                                                                                                                                                                           
 - 64 rows × 16 slots                                                                                                                                                                                           
 - 64 rows × 32 slots                                                                                                                                                                                           
 - fewer register accumulators with more dispatch parallelism                                                                                                                                                   
                                                                                                                                                                                                                
 The goal is to balance LDS reuse against register occupancy. MoE GEMMs are currently the largest combined bottleneck.                                                                                          
                                                                                                                                                                                                                
 ### 6. Reduce host synchronization during prefill                                                                                                                                                              
                                                                                                                                                                                                                
 The profiler showed multiple tensor-read waits and many eager submissions. Investigate whether prompt processing can keep these operations GPU-resident:                                                       
                                                                                                                                                                                                                
 - router/top-k results;                                                                                                                                                                                        
 - expert tile construction;                                                                                                                                                                                    
 - attention softmax;                                                                                                                                                                                           
 - KV metadata;                                                                                                                                                                                                 
 - intermediate activation transformations.                                                                                                                                                                     
                                                                                                                                                                                                                
 Even if individual kernels are unchanged, removing host readbacks could improve end-to-end PP.                                                                                                                 
                                                                                                                                                                                                                
 ### 7. Add a GPU-resident router-to-MoE path                                                                                                                                                                   
                                                                                                                                                                                                                
 The current sequence appears to retain some host orchestration around routing. A fully GPU-resident path could chain:                                                                                          
                                                                                                                                                                                                                
 ```text                                                                                                                                                                                                        
   router → top-k → tile construction → gate/up GEMM → down GEMM → reduce                                                                                                                                       
 ```                                                                                                                                                                                                            
                                                                                                                                                                                                                
 This is likely more valuable than optimizing router_topk itself.                                                                                                                                               
                                                                                                                                                                                                                
 ### 8. Optimize weight layout for Phoenix                                                                                                                                                                      
                                                                                                                                                                                                                
 Investigate prepacking weights into layouts specifically suited to:                                                                                                                                            
                                                                                                                                                                                                                
 - 128-bit/coalesced loads;                                                                                                                                                                                     
 - 64-lane waves;                                                                                                                                                                                               
 - cooperative matrices;                                                                                                                                                                                        
 - integer dot products.                                                                                                                                                                                        
                                                                                                                                                                                                                
 The current shaders repeatedly reconstruct/dequantize weights inside each workgroup. A one-time GPU-native packed representation may increase memory use but reduce repeated unpacking.                        
                                                                                                                                                                                                                
 ### 9. Test 64-lane delta decode separately                                                                                                                                                                    
                                                                                                                                                                                                                
 For decode rather than PP, the current delta_net_decode uses 32 threads while the hardware subgroup is 64. A 64-column variant may improve wave utilization, though this is unlikely to improve prompt         
 throughput substantially unless recurrent prefill uses the same path.                                                                                                                                          
                                                                                                                                                                                                                
 ### 10. Measure memory-clock behavior                                                                                                                                                                          
                                                                                                                                                                                                                
 The GPU core is confirmed at 2700 MHz, but Phoenix performance can still be limited by shared-memory bandwidth. Record during benchmarks:                                                                      
                                                                                                                                                                                                                
 ```sh                                                                                                                                                                                                          
   watch -n .2 '                                                                                                                                                                                                
   cat /sys/class/drm/card1/device/hwmon/hwmon3/freq1_input                                                                                                                                                     
   cat /sys/class/drm/card1/device/gpu_busy_percent                                                                                                                                                             
   '                                                                                                                                                                                                            
 ```                                                                                                                                                                                                            
                                                                                                                                                                                                                
 If GPU busy is high but throughput scales poorly with clock, prioritize bandwidth/layout work. If GPU busy is low, prioritize dispatch batching, synchronization, and occupancy.                               
                                                                                                                                                                                                                
 The highest-upside path is:                                                                                                                                                                                    
                                                                                                                                                                                                                
 1. prefill chunk sweep;                                                                                                                                                                                        
 2. cooperative-matrix/int8-dot prototype;                                                                                                                                                                      
 3. MoE tile tuning;                                                                                                                                                                                            
 4. GPU-resident routing and synchronization reduction;                                                                                                                                                         
 5. quantization-format comparison.  

---
## Measurement log (2024-08-08): tile-shape round

All GEMM/decode geometry alternatives below were built, benchmarked on-device
(2048 ctx, same command as `achievements.md`), and reverted. Baseline:
prefill **208.90 t/s**, gen **24.23 t/s**.

- Dense Q8 GEMM 128x128 (8x8 regs): 188.82 (−10%) — register pressure.
- Dense Q8 GEMM 64x64 f32: 176.73 (−15%) — f16 128x64 stays.
- MoE 64-slot 512-thread tiles: 183.19 (−12%) — 32-slot stays.
- Delta decode 64-col: 162.58 pref / 18.49 gen (−22%) — 32-col stays.
- `Q36_VK_DELTA_DECODE=0` (fast path): 150.22 (−28%) — decode kernel stays.
- `Q36_VK_DELTA_COL_PREFILL=1` at 2048: 165.50 (−21%) — Phoenix default
  (disabled) confirmed at long context, not just the 512-token probe.

Conclusion: items 5 and 9 are now closed with measurements (keep current
tiles / keep 32-col decode). Item 2 (cooperative matrix) is the remaining
open item: runtime shape list confirmed (16x16x16 F16x F16→F32 exists),
but GLSL tooling cannot emit it (glslang lacks GL_KHR_cooperative_matrix),
so a prototype requires hand-written SPIR-V.

Next candidates to attack first (ranked by measured GPU share and risk):

1. Cooperative-matrix GEMM prototype (SPIR-V assembly) for dense Q8 + MoE
   gate/up (25.7% + 25.3% of profiled time).
2. Activation-quantized int8 dot product path for dense Q8 (needs a new
   activation quantization kernel upstream of matmul; A8W8 dot is exposed
   by the device).
3. Phoenix attention variant of `attn_prefill_qtile2` (11.9% today, grows
   with context; fp32 score/value chains are the likely cost).
4. Memory-clock sampling during benchmarks (`watch` sysfs hwmon) to check
   whether the shared-DDR5 ceiling is hit at higher context lengths.
