/* SFU-Sim: bit-exact model of the NVIDIA MUFU special-function unit.
 *
 * Link with:  -Llib -lmufu_model      (Linux, lib/libmufu_model.a)
 *             lib\mufu_model.lib      (Windows, MSVC)
 *
 * Every function maps an input bit pattern to the output bit pattern produced by the bare MUFU
 * instruction. The functions are pure and thread-safe.
 *
 *   f32 functions   : 32-bit IEEE-754 single bit pattern in, 32-bit pattern out.
 *                     sin / cos take their argument in turns (the PTX layer multiplies by 1/(2*pi)
 *                     before the instruction, see ptx/minimal.hpp).
 *   64h functions   : the argument and the result are the HIGH 32 bits of an IEEE-754 double.
 *   f16 / bf16      : 16-bit pattern in, 16-bit pattern out.
 */
#ifndef MUFU_MODEL_H
#define MUFU_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* sfu_model_version(void);

uint32_t sfu_mufu_ex2(uint32_t bits);
uint32_t sfu_mufu_lg2(uint32_t bits);
uint32_t sfu_mufu_rcp(uint32_t bits);
uint32_t sfu_mufu_rsq(uint32_t bits);
uint32_t sfu_mufu_sqrt(uint32_t bits);
uint32_t sfu_mufu_tanh(uint32_t bits);
uint32_t sfu_mufu_sin(uint32_t bits_in_turns);
uint32_t sfu_mufu_cos(uint32_t bits_in_turns);

uint32_t sfu_mufu_rcp64h(uint32_t high_word);
uint32_t sfu_mufu_rsq64h(uint32_t high_word);

uint16_t sfu_mufu_ex2_f16(uint16_t bits);
uint16_t sfu_mufu_tanh_f16(uint16_t bits);
uint16_t sfu_mufu_ex2_bf16(uint16_t bits);
uint16_t sfu_mufu_tanh_bf16(uint16_t bits);

#ifdef __cplusplus
}
#endif

#endif /* MUFU_MODEL_H */
