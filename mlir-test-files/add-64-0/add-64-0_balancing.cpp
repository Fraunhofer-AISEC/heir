
#include "openfhe/pke/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using ConstCiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextBGVRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

std::vector<CiphertextT> func(CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1, std::vector<CiphertextT> v2, std::vector<CiphertextT> v3, std::vector<CiphertextT> v4, std::vector<CiphertextT> v5, std::vector<CiphertextT> v6, std::vector<CiphertextT> v7, std::vector<CiphertextT> v8, std::vector<CiphertextT> v9, std::vector<CiphertextT> v10, std::vector<CiphertextT> v11, std::vector<CiphertextT> v12, std::vector<CiphertextT> v13, std::vector<CiphertextT> v14, std::vector<CiphertextT> v15, std::vector<CiphertextT> v16, std::vector<CiphertextT> v17, std::vector<CiphertextT> v18, std::vector<CiphertextT> v19, std::vector<CiphertextT> v20, std::vector<CiphertextT> v21, std::vector<CiphertextT> v22, std::vector<CiphertextT> v23, std::vector<CiphertextT> v24, std::vector<CiphertextT> v25, std::vector<CiphertextT> v26, std::vector<CiphertextT> v27, std::vector<CiphertextT> v28, std::vector<CiphertextT> v29, std::vector<CiphertextT> v30, std::vector<CiphertextT> v31, std::vector<CiphertextT> v32, std::vector<CiphertextT> v33, std::vector<CiphertextT> v34, std::vector<CiphertextT> v35, std::vector<CiphertextT> v36, std::vector<CiphertextT> v37, std::vector<CiphertextT> v38, std::vector<CiphertextT> v39, std::vector<CiphertextT> v40, std::vector<CiphertextT> v41, std::vector<CiphertextT> v42, std::vector<CiphertextT> v43, std::vector<CiphertextT> v44, std::vector<CiphertextT> v45, std::vector<CiphertextT> v46, std::vector<CiphertextT> v47, std::vector<CiphertextT> v48, std::vector<CiphertextT> v49, std::vector<CiphertextT> v50, std::vector<CiphertextT> v51, std::vector<CiphertextT> v52, std::vector<CiphertextT> v53, std::vector<CiphertextT> v54, std::vector<CiphertextT> v55, std::vector<CiphertextT> v56, std::vector<CiphertextT> v57, std::vector<CiphertextT> v58, std::vector<CiphertextT> v59, std::vector<CiphertextT> v60, std::vector<CiphertextT> v61, std::vector<CiphertextT> v62, std::vector<CiphertextT> v63) {
  [[maybe_unused]] size_t v64 = 0;
  auto ct = v0[0];
  const auto& ct1 = v1[0];
  cc->EvalAddInPlace(ct, ct1);
  auto ct3 = v2[0];
  const auto& ct4 = v3[0];
  cc->EvalAddInPlace(ct3, ct4);
  cc->EvalAddInPlace(ct, ct3);
  auto ct7 = v4[0];
  const auto& ct8 = v5[0];
  cc->EvalAddInPlace(ct7, ct8);
  auto ct10 = v6[0];
  const auto& ct11 = v7[0];
  cc->EvalAddInPlace(ct10, ct11);
  cc->EvalAddInPlace(ct7, ct10);
  cc->EvalAddInPlace(ct, ct7);
  auto ct15 = v8[0];
  const auto& ct16 = v9[0];
  cc->EvalAddInPlace(ct15, ct16);
  auto ct18 = v10[0];
  const auto& ct19 = v11[0];
  cc->EvalAddInPlace(ct18, ct19);
  cc->EvalAddInPlace(ct15, ct18);
  auto ct22 = v12[0];
  const auto& ct23 = v13[0];
  cc->EvalAddInPlace(ct22, ct23);
  auto ct25 = v14[0];
  const auto& ct26 = v15[0];
  cc->EvalAddInPlace(ct25, ct26);
  cc->EvalAddInPlace(ct22, ct25);
  cc->EvalAddInPlace(ct15, ct22);
  cc->EvalAddInPlace(ct, ct15);
  auto ct31 = v16[0];
  const auto& ct32 = v17[0];
  cc->EvalAddInPlace(ct31, ct32);
  auto ct34 = v18[0];
  const auto& ct35 = v19[0];
  cc->EvalAddInPlace(ct34, ct35);
  cc->EvalAddInPlace(ct31, ct34);
  auto ct38 = v20[0];
  const auto& ct39 = v21[0];
  cc->EvalAddInPlace(ct38, ct39);
  auto ct41 = v22[0];
  const auto& ct42 = v23[0];
  cc->EvalAddInPlace(ct41, ct42);
  cc->EvalAddInPlace(ct38, ct41);
  cc->EvalAddInPlace(ct31, ct38);
  auto ct46 = v24[0];
  const auto& ct47 = v25[0];
  cc->EvalAddInPlace(ct46, ct47);
  auto ct49 = v26[0];
  const auto& ct50 = v27[0];
  cc->EvalAddInPlace(ct49, ct50);
  cc->EvalAddInPlace(ct46, ct49);
  auto ct53 = v28[0];
  const auto& ct54 = v29[0];
  cc->EvalAddInPlace(ct53, ct54);
  auto ct56 = v30[0];
  const auto& ct57 = v31[0];
  cc->EvalAddInPlace(ct56, ct57);
  cc->EvalAddInPlace(ct53, ct56);
  cc->EvalAddInPlace(ct46, ct53);
  cc->EvalAddInPlace(ct31, ct46);
  cc->EvalAddInPlace(ct, ct31);
  auto ct63 = v32[0];
  const auto& ct64 = v33[0];
  cc->EvalAddInPlace(ct63, ct64);
  auto ct66 = v34[0];
  const auto& ct67 = v35[0];
  cc->EvalAddInPlace(ct66, ct67);
  cc->EvalAddInPlace(ct63, ct66);
  auto ct70 = v36[0];
  const auto& ct71 = v37[0];
  cc->EvalAddInPlace(ct70, ct71);
  auto ct73 = v38[0];
  const auto& ct74 = v39[0];
  cc->EvalAddInPlace(ct73, ct74);
  cc->EvalAddInPlace(ct70, ct73);
  cc->EvalAddInPlace(ct63, ct70);
  auto ct78 = v40[0];
  const auto& ct79 = v41[0];
  cc->EvalAddInPlace(ct78, ct79);
  auto ct81 = v42[0];
  const auto& ct82 = v43[0];
  cc->EvalAddInPlace(ct81, ct82);
  cc->EvalAddInPlace(ct78, ct81);
  auto ct85 = v44[0];
  const auto& ct86 = v45[0];
  cc->EvalAddInPlace(ct85, ct86);
  auto ct88 = v46[0];
  const auto& ct89 = v47[0];
  cc->EvalAddInPlace(ct88, ct89);
  cc->EvalAddInPlace(ct85, ct88);
  cc->EvalAddInPlace(ct78, ct85);
  cc->EvalAddInPlace(ct63, ct78);
  auto ct94 = v48[0];
  const auto& ct95 = v49[0];
  cc->EvalAddInPlace(ct94, ct95);
  auto ct97 = v50[0];
  const auto& ct98 = v51[0];
  cc->EvalAddInPlace(ct97, ct98);
  cc->EvalAddInPlace(ct94, ct97);
  auto ct101 = v52[0];
  const auto& ct102 = v53[0];
  cc->EvalAddInPlace(ct101, ct102);
  auto ct104 = v54[0];
  const auto& ct105 = v55[0];
  cc->EvalAddInPlace(ct104, ct105);
  cc->EvalAddInPlace(ct101, ct104);
  cc->EvalAddInPlace(ct94, ct101);
  auto ct109 = v56[0];
  const auto& ct110 = v57[0];
  cc->EvalAddInPlace(ct109, ct110);
  auto ct112 = v58[0];
  const auto& ct113 = v59[0];
  cc->EvalAddInPlace(ct112, ct113);
  cc->EvalAddInPlace(ct109, ct112);
  auto ct116 = v60[0];
  const auto& ct117 = v61[0];
  cc->EvalAddInPlace(ct116, ct117);
  auto ct119 = v62[0];
  const auto& ct120 = v63[0];
  cc->EvalAddInPlace(ct119, ct120);
  cc->EvalAddInPlace(ct116, ct119);
  cc->EvalAddInPlace(ct109, ct116);
  cc->EvalAddInPlace(ct94, ct109);
  cc->EvalAddInPlace(ct63, ct94);
  cc->EvalAddInPlace(ct, ct63);
  auto ct127 = cc->EvalMultNoRelin(ct, ct);
  cc->RelinearizeInPlace(ct127);
  cc->ModReduceInPlace(ct127);
  auto ct130 = cc->EvalMultNoRelin(ct127, ct127);
  cc->RelinearizeInPlace(ct130);
  cc->ModReduceInPlace(ct130);
  auto ct133 = cc->EvalMultNoRelin(ct130, ct130);
  cc->RelinearizeInPlace(ct133);
  cc->ModReduceInPlace(ct133);
  auto ct136 = cc->EvalMultNoRelin(ct133, ct133);
  cc->RelinearizeInPlace(ct136);
  std::vector<CiphertextT> v65(1);
  cc->ModReduceInPlace(ct136);
  v65[0] = ct136;
  return v65;
}
std::vector<CiphertextT> func__encrypt__arg0(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg1(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg2(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg3(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg4(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg5(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg6(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg7(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg8(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg9(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg10(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg11(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg12(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg13(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg14(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg15(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg16(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg17(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg18(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg19(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg20(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg21(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg22(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg23(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg24(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg25(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg26(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg27(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg28(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg29(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg30(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg31(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg32(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg33(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg34(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg35(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg36(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg37(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg38(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg39(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg40(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg41(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg42(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg43(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg44(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg45(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg46(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg47(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg48(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg49(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg50(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg51(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg52(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg53(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg54(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg55(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg56(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg57(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg58(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg59(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg60(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg61(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg62(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> func__encrypt__arg63(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 1024;
  [[maybe_unused]] int32_t v2 = 1;
  [[maybe_unused]] int32_t v3 = 0;
  std::vector<int16_t> v4(1024, 0);
  [[maybe_unused]] int32_t v5 = 8;
  [[maybe_unused]] size_t v6 = 0;
  std::vector<int16_t> v7 = v4;
  for (auto v8 = 0; v8 < 1024; ++v8) {
    int32_t v10 = v8 % v5;
    size_t v11 = static_cast<size_t>(v10);
    int16_t v12 = v0[v11];
    size_t v13 = static_cast<size_t>(v8);
    v7[v13 + 1024 * (0)] = v12;
  }
  std::vector<int16_t> v15(1024);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 1024; ++v15_i1) {
      v15[v15_i1 + 1024 * (v15_i0)] = v7[0 + v15_i1 * 1 + 1024 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<int16_t> func__decrypt__result0(CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  std::vector<int16_t> v1(8, 0);
  [[maybe_unused]] int32_t v2 = 0;
  [[maybe_unused]] int32_t v3 = 1;
  [[maybe_unused]] int32_t v4 = 8;
  [[maybe_unused]] int32_t v5 = 1024;
  [[maybe_unused]] size_t v6 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(1024);
  const auto& v7_cast = pt->GetPackedValue();
  std::vector<int16_t> v7(std::begin(v7_cast), std::end(v7_cast));
  std::vector<int16_t> v8 = v1;
  for (auto v9 = 0; v9 < 1024; ++v9) {
    int32_t v11 = v9 % v4;
    size_t v12 = static_cast<size_t>(v9);
    int16_t v13 = v7[v12 + 1024 * (0)];
    size_t v14 = static_cast<size_t>(v11);
    v8[v14] = v13;
  }
  return v8;
}
CryptoContextT func__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(4);
  params.SetPlaintextModulus(786433);
  params.SetRingDim(16384);
  params.SetFirstModSize(37);
  params.SetScalingModSize(54);
  params.SetKeySwitchTechnique(HYBRID);
  params.SetScalingTechnique(FIXEDMANUAL);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT func__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  return cc;
}
