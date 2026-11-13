#include <cstdint>
#include <iostream>
#include <vector>

#include "form-increasing_direct.h"
#include "../params.h"
#include "src/pke/include/openfhe.h"

using namespace lbcrypto;

void decrypt_and_print(CryptoContext<DCRTPoly> cc,
                       const Ciphertext<DCRTPoly>& ct,
                       PrivateKey<DCRTPoly> sk,
                       const std::string& label,
                       int16_t expected) {
  Plaintext pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(1024);
  const auto& vals = pt->GetPackedValue();
  std::cerr << label << ": [";
  for (size_t i = 0; i < 8; i++) {
    int16_t v = static_cast<int16_t>(vals[i]);
    std::cerr << v;
    if (i < 7) std::cerr << ", ";
  }
  std::cerr << "] (expected " << expected << ")";
  std::cerr << " towers=" << ct->GetElements()[0].GetNumOfElements();
  std::cerr << std::endl;
}

int main() {
  auto cc = func__generate_crypto_context();
  auto keyPair = cc->KeyGen();
  cc = func__configure_crypto_context(cc, keyPair.secretKey);
  printModulusChain(cc, "form-increasing", "direct");

  std::vector<std::vector<int16_t>> args(64);
  for (int i = 0; i < 64; i++) {
    args[i] = std::vector<int16_t>(8, i == 0 ? 1 : 0);
  }

  std::vector<std::vector<Ciphertext<DCRTPoly>>> enc(64);
  enc[0] = func__encrypt__arg0(cc, args[0], keyPair.publicKey);
  for (int i = 1; i < 64; i++) {
    enc[i] = func__encrypt__arg0(cc, args[i], keyPair.publicKey);
  }
  auto sk = keyPair.secretKey;

  // === Match generated code: m1 = arg0*arg0 (no ModReduce) ===
  auto m1 = cc->EvalMultNoRelin(enc[0][0], enc[0][0]);
  cc->RelinearizeInPlace(m1);
  decrypt_and_print(cc, m1, sk, "m1 = arg0^2", 1);

  // === Level 2: square ALL 16 args at level 0, add all, THEN ModReduce ===
  std::vector<Ciphertext<DCRTPoly>> sq2(16);
  for (int i = 0; i < 16; i++) {
    sq2[i] = cc->EvalMultNoRelin(enc[i][0], enc[i][0]);
    cc->RelinearizeInPlace(sq2[i]);
  }
  // Add: acc = m1 + sq2[0] + sq2[1] + ... + sq2[15]
  // Generated code does: ct64 = m1; EvalAddInPlace(ct64, ct64) = m1 + m1
  // (because CSE merged m1 and s2_0 = arg0^2)
  auto acc = m1;
  cc->EvalAddInPlace(acc, sq2[0]); // m1 + arg0^2 = 1 + 1 = 2 (but generated does m1+m1)
  // Wait - generated code does EvalAddInPlace(ct64, ct64) which is m1+m1
  // After CSE, s2_0 = m1, so l2_1 = m1 + s2_0 = m1 + m1
  // Let me match exactly:
  // acc = m1 + m1 + sq2[1] + ... + sq2[15]
  // But sq2[0] = arg0^2 = m1, so acc = m1 + sq2[0] + sq2[1] + ... is the same
  
  for (int i = 1; i < 16; i++) {
    cc->EvalAddInPlace(acc, sq2[i]);
  }
  // Now ModReduce the accumulator
  cc->ModReduceInPlace(acc);
  decrypt_and_print(cc, acc, sk, "L2 acc after ModReduce", 2);

  // Barrier mul m2 = acc^2
  auto m2 = cc->EvalMultNoRelin(acc, acc);
  cc->RelinearizeInPlace(m2);
  // Generated code does NOT ModReduce m2 here — it stays at level 1
  // But m2 needs ModReduce before Level 3 adds
  // Actually let me check: does generated code ModReduce m2?
  // From structure: ModReduce #2 at line 391 is for v69 (m1), NOT for m2
  // So m2 (v265) stays at level 1 (4 towers)
  decrypt_and_print(cc, m2, sk, "m2 = L2^2 (no ModReduce)", 4);

  // === Level 3: reuse m1 (arg0^2) by ModReducing it separately ===
  // Generated code: v269 = ModReduce(v69=m1), then v273 = v269^2 = arg0^4
  // For all 32 args: ModReduce their Level 2 squarings, then square again
  
  // First, ModReduce m1 for reuse as s3_0_1
  auto s3_0 = m1;  // copy
  cc->ModReduceInPlace(s3_0);  // arg0^2 at level 1
  auto s3_0_sq = cc->EvalMultNoRelin(s3_0, s3_0);  // arg0^4 at level 1
  cc->RelinearizeInPlace(s3_0_sq);
  
  // For args 1-15: reuse their Level 2 squarings (sq2[1..15])
  // ModReduce them, then square
  std::vector<Ciphertext<DCRTPoly>> sq3(32);
  sq3[0] = s3_0_sq;
  for (int i = 1; i < 16; i++) {
    auto mr = sq2[i];  // copy
    cc->ModReduceInPlace(mr);  // arg_i^2 at level 1
    sq3[i] = cc->EvalMultNoRelin(mr, mr);  // arg_i^4 at level 1
    cc->RelinearizeInPlace(sq3[i]);
  }
  // For args 16-31: these weren't squared in Level 2, so square fresh
  for (int i = 16; i < 32; i++) {
    auto sq1 = cc->EvalMultNoRelin(enc[i][0], enc[i][0]);
    cc->RelinearizeInPlace(sq1);
    cc->ModReduceInPlace(sq1);
    sq3[i] = cc->EvalMultNoRelin(sq1, sq1);
    cc->RelinearizeInPlace(sq3[i]);
    cc->ModReduceInPlace(sq3[i]);
  }
  // Wait - args 16-31 in the generated code are NOT fresh squarings.
  // The generated code squares them once (at level 0), then ModReduces and squares again.
  // But they weren't squared in Level 2, so the generated code must square them fresh.
  // Actually, looking at the MLIR: Level 3 squares args 0..31, each twice.
  // Args 0..15 were already squared in Level 2 (reused via CSE).
  // Args 16..31 are squared fresh for Level 3.
  // But wait - the generated code structure shows:
  // Before ModReduce #3-#17: 1 mul each → these are the 16 Level 2 arg squarings being ModReduced and squared again
  // Before ModReduce #18-#33: 2 muls each → these are args 16-31 being squared twice (fresh)
  
  // Hmm, this is getting complicated. Let me just add to m2:
  acc = m2;
  for (int i = 0; i < 32; i++) {
    cc->EvalAddInPlace(acc, sq3[i]);
  }
  cc->ModReduceInPlace(acc);
  decrypt_and_print(cc, acc, sk, "L3 acc after ModReduce", 5);

  // Barrier mul m3 = acc^2
  auto m3 = cc->EvalMultNoRelin(acc, acc);
  cc->RelinearizeInPlace(m3);
  decrypt_and_print(cc, m3, sk, "m3 = L3^2", 25);

  // === Level 4: 48 args, each squared 3 times ===
  // Reuse: arg0^4 (from Level 3), square to get arg0^8
  // Args 0..31: reuse their Level 3 squarings (sq3[0..31])
  // Args 32..47: fresh squarings
  std::vector<Ciphertext<DCRTPoly>> sq4(48);
  for (int i = 0; i < 32; i++) {
    auto mr = sq3[i];  // copy
    cc->ModReduceInPlace(mr);  // arg_i^4 at level 2
    sq4[i] = cc->EvalMultNoRelin(mr, mr);  // arg_i^8 at level 2
    cc->RelinearizeInPlace(sq4[i]);
    cc->ModReduceInPlace(sq4[i]);  // to level 3
  }
  for (int i = 32; i < 48; i++) {
    auto sq1 = cc->EvalMultNoRelin(enc[i][0], enc[i][0]);
    cc->RelinearizeInPlace(sq1);
    cc->ModReduceInPlace(sq1);
    auto sq2_ = cc->EvalMultNoRelin(sq1, sq1);
    cc->RelinearizeInPlace(sq2_);
    cc->ModReduceInPlace(sq2_);
    sq4[i] = cc->EvalMultNoRelin(sq2_, sq2_);
    cc->RelinearizeInPlace(sq4[i]);
    cc->ModReduceInPlace(sq4[i]);
  }
  acc = m3;
  // ModReduce m3 to match sq4 level
  cc->ModReduceInPlace(acc);  // m3 to level 3
  for (int i = 0; i < 48; i++) {
    cc->EvalAddInPlace(acc, sq4[i]);
  }
  cc->ModReduceInPlace(acc);
  decrypt_and_print(cc, acc, sk, "L4 acc after ModReduce", 26);

  // Barrier mul m4 = acc^2
  auto m4 = cc->EvalMultNoRelin(acc, acc);
  cc->RelinearizeInPlace(m4);
  decrypt_and_print(cc, m4, sk, "m4 = L4^2", 676);

  // === Level 5: 64 args, each squared 4 times ===
  std::vector<Ciphertext<DCRTPoly>> sq5(64);
  for (int i = 0; i < 48; i++) {
    auto mr = sq4[i];
    cc->ModReduceInPlace(mr);
    sq5[i] = cc->EvalMultNoRelin(mr, mr);
    cc->RelinearizeInPlace(sq5[i]);
    cc->ModReduceInPlace(sq5[i]);
  }
  for (int i = 48; i < 64; i++) {
    auto sq1 = cc->EvalMultNoRelin(enc[i][0], enc[i][0]);
    cc->RelinearizeInPlace(sq1);
    cc->ModReduceInPlace(sq1);
    auto sq2_ = cc->EvalMultNoRelin(sq1, sq1);
    cc->RelinearizeInPlace(sq2_);
    cc->ModReduceInPlace(sq2_);
    auto sq3_ = cc->EvalMultNoRelin(sq2_, sq2_);
    cc->RelinearizeInPlace(sq3_);
    cc->ModReduceInPlace(sq3_);
    sq5[i] = cc->EvalMultNoRelin(sq3_, sq3_);
    cc->RelinearizeInPlace(sq5[i]);
    cc->ModReduceInPlace(sq5[i]);
  }
  acc = m4;
  cc->ModReduceInPlace(acc);
  for (int i = 0; i < 64; i++) {
    cc->EvalAddInPlace(acc, sq5[i]);
  }
  decrypt_and_print(cc, acc, sk, "L5 final", 677);

  return 0;
}
