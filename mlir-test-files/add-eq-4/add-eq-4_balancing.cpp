
#include "openfhe/pke/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using MutableCiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextBGVRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT func(CryptoContextT cc, CiphertextT ct, CiphertextT ct1, CiphertextT ct2, CiphertextT ct3, CiphertextT ct4, CiphertextT ct5, CiphertextT ct6, CiphertextT ct7, CiphertextT ct8, CiphertextT ct9, CiphertextT ct10, CiphertextT ct11, CiphertextT ct12, CiphertextT ct13, CiphertextT ct14, CiphertextT ct15, CiphertextT ct16, CiphertextT ct17, CiphertextT ct18, CiphertextT ct19, CiphertextT ct20, CiphertextT ct21, CiphertextT ct22, CiphertextT ct23, CiphertextT ct24, CiphertextT ct25, CiphertextT ct26, CiphertextT ct27, CiphertextT ct28, CiphertextT ct29, CiphertextT ct30, CiphertextT ct31, CiphertextT ct32, CiphertextT ct33, CiphertextT ct34, CiphertextT ct35, CiphertextT ct36, CiphertextT ct37, CiphertextT ct38, CiphertextT ct39, CiphertextT ct40, CiphertextT ct41, CiphertextT ct42, CiphertextT ct43, CiphertextT ct44, CiphertextT ct45, CiphertextT ct46, CiphertextT ct47, CiphertextT ct48, CiphertextT ct49, CiphertextT ct50, CiphertextT ct51, CiphertextT ct52, CiphertextT ct53, CiphertextT ct54, CiphertextT ct55, CiphertextT ct56, CiphertextT ct57, CiphertextT ct58, CiphertextT ct59, CiphertextT ct60, CiphertextT ct61, CiphertextT ct62, CiphertextT ct63) {
  const auto& ct64 = cc->EvalMultNoRelin(ct, ct);
  const auto& ct65 = cc->Relinearize(ct64);
  const auto& ct66 = cc->EvalMultNoRelin(ct1, ct1);
  const auto& ct67 = cc->Relinearize(ct66);
  const auto& ct68 = cc->EvalMultNoRelin(ct2, ct2);
  const auto& ct69 = cc->Relinearize(ct68);
  const auto& ct70 = cc->EvalMultNoRelin(ct3, ct3);
  const auto& ct71 = cc->Relinearize(ct70);
  const auto& ct72 = cc->EvalMultNoRelin(ct4, ct4);
  const auto& ct73 = cc->Relinearize(ct72);
  const auto& ct74 = cc->EvalMultNoRelin(ct5, ct5);
  const auto& ct75 = cc->Relinearize(ct74);
  const auto& ct76 = cc->EvalMultNoRelin(ct6, ct6);
  const auto& ct77 = cc->Relinearize(ct76);
  const auto& ct78 = cc->EvalMultNoRelin(ct7, ct7);
  const auto& ct79 = cc->Relinearize(ct78);
  const auto& ct80 = cc->EvalMultNoRelin(ct8, ct8);
  const auto& ct81 = cc->Relinearize(ct80);
  const auto& ct82 = cc->EvalMultNoRelin(ct9, ct9);
  const auto& ct83 = cc->Relinearize(ct82);
  const auto& ct84 = cc->EvalMultNoRelin(ct10, ct10);
  const auto& ct85 = cc->Relinearize(ct84);
  const auto& ct86 = cc->EvalMultNoRelin(ct11, ct11);
  const auto& ct87 = cc->Relinearize(ct86);
  const auto& ct88 = cc->EvalMultNoRelin(ct12, ct12);
  const auto& ct89 = cc->Relinearize(ct88);
  const auto& ct90 = cc->EvalMultNoRelin(ct13, ct13);
  const auto& ct91 = cc->Relinearize(ct90);
  const auto& ct92 = cc->EvalMultNoRelin(ct14, ct14);
  const auto& ct93 = cc->Relinearize(ct92);
  const auto& ct94 = cc->EvalMultNoRelin(ct15, ct15);
  const auto& ct95 = cc->Relinearize(ct94);
  const auto& ct96 = cc->EvalMultNoRelin(ct16, ct16);
  const auto& ct97 = cc->Relinearize(ct96);
  const auto& ct98 = cc->EvalMultNoRelin(ct17, ct17);
  const auto& ct99 = cc->Relinearize(ct98);
  const auto& ct100 = cc->EvalMultNoRelin(ct18, ct18);
  const auto& ct101 = cc->Relinearize(ct100);
  const auto& ct102 = cc->EvalMultNoRelin(ct19, ct19);
  const auto& ct103 = cc->Relinearize(ct102);
  const auto& ct104 = cc->EvalMultNoRelin(ct20, ct20);
  const auto& ct105 = cc->Relinearize(ct104);
  const auto& ct106 = cc->EvalMultNoRelin(ct21, ct21);
  const auto& ct107 = cc->Relinearize(ct106);
  const auto& ct108 = cc->EvalMultNoRelin(ct22, ct22);
  const auto& ct109 = cc->Relinearize(ct108);
  const auto& ct110 = cc->EvalMultNoRelin(ct23, ct23);
  const auto& ct111 = cc->Relinearize(ct110);
  const auto& ct112 = cc->EvalMultNoRelin(ct24, ct24);
  const auto& ct113 = cc->Relinearize(ct112);
  const auto& ct114 = cc->EvalMultNoRelin(ct25, ct25);
  const auto& ct115 = cc->Relinearize(ct114);
  const auto& ct116 = cc->EvalMultNoRelin(ct26, ct26);
  const auto& ct117 = cc->Relinearize(ct116);
  const auto& ct118 = cc->EvalMultNoRelin(ct27, ct27);
  const auto& ct119 = cc->Relinearize(ct118);
  const auto& ct120 = cc->EvalMultNoRelin(ct28, ct28);
  const auto& ct121 = cc->Relinearize(ct120);
  const auto& ct122 = cc->EvalMultNoRelin(ct29, ct29);
  const auto& ct123 = cc->Relinearize(ct122);
  const auto& ct124 = cc->EvalMultNoRelin(ct30, ct30);
  const auto& ct125 = cc->Relinearize(ct124);
  const auto& ct126 = cc->EvalMultNoRelin(ct31, ct31);
  const auto& ct127 = cc->Relinearize(ct126);
  const auto& ct128 = cc->EvalAdd(ct65, ct65);
  const auto& ct129 = cc->EvalAdd(ct67, ct69);
  const auto& ct130 = cc->EvalAdd(ct128, ct129);
  const auto& ct131 = cc->EvalAdd(ct71, ct73);
  const auto& ct132 = cc->EvalAdd(ct75, ct77);
  const auto& ct133 = cc->EvalAdd(ct131, ct132);
  const auto& ct134 = cc->EvalAdd(ct130, ct133);
  const auto& ct135 = cc->EvalAdd(ct79, ct81);
  const auto& ct136 = cc->EvalAdd(ct83, ct85);
  const auto& ct137 = cc->EvalAdd(ct135, ct136);
  const auto& ct138 = cc->EvalAdd(ct87, ct89);
  const auto& ct139 = cc->EvalAdd(ct91, ct93);
  const auto& ct140 = cc->EvalAdd(ct138, ct139);
  const auto& ct141 = cc->EvalAdd(ct137, ct140);
  const auto& ct142 = cc->EvalAdd(ct134, ct141);
  const auto& ct143 = cc->EvalAdd(ct95, ct97);
  const auto& ct144 = cc->EvalAdd(ct99, ct101);
  const auto& ct145 = cc->EvalAdd(ct143, ct144);
  const auto& ct146 = cc->EvalAdd(ct103, ct105);
  const auto& ct147 = cc->EvalAdd(ct107, ct109);
  const auto& ct148 = cc->EvalAdd(ct146, ct147);
  const auto& ct149 = cc->EvalAdd(ct145, ct148);
  const auto& ct150 = cc->EvalAdd(ct111, ct113);
  const auto& ct151 = cc->EvalAdd(ct115, ct117);
  const auto& ct152 = cc->EvalAdd(ct150, ct151);
  const auto& ct153 = cc->EvalAdd(ct119, ct121);
  const auto& ct154 = cc->EvalAdd(ct123, ct125);
  const auto& ct155 = cc->EvalAdd(ct154, ct127);
  const auto& ct156 = cc->EvalAdd(ct153, ct155);
  const auto& ct157 = cc->EvalAdd(ct152, ct156);
  const auto& ct158 = cc->EvalAdd(ct149, ct157);
  const auto& ct159 = cc->EvalAdd(ct142, ct158);
  const auto& ct160 = cc->ModReduce(ct159);
  const auto& ct161 = cc->EvalMultNoRelin(ct160, ct160);
  const auto& ct162 = cc->Relinearize(ct161);
  const auto& ct163 = cc->ModReduce(ct65);
  const auto& ct164 = cc->EvalMultNoRelin(ct163, ct163);
  const auto& ct165 = cc->Relinearize(ct164);
  const auto& ct166 = cc->ModReduce(ct67);
  const auto& ct167 = cc->EvalMultNoRelin(ct166, ct166);
  const auto& ct168 = cc->Relinearize(ct167);
  const auto& ct169 = cc->ModReduce(ct69);
  const auto& ct170 = cc->EvalMultNoRelin(ct169, ct169);
  const auto& ct171 = cc->Relinearize(ct170);
  const auto& ct172 = cc->ModReduce(ct71);
  const auto& ct173 = cc->EvalMultNoRelin(ct172, ct172);
  const auto& ct174 = cc->Relinearize(ct173);
  const auto& ct175 = cc->ModReduce(ct73);
  const auto& ct176 = cc->EvalMultNoRelin(ct175, ct175);
  const auto& ct177 = cc->Relinearize(ct176);
  const auto& ct178 = cc->ModReduce(ct75);
  const auto& ct179 = cc->EvalMultNoRelin(ct178, ct178);
  const auto& ct180 = cc->Relinearize(ct179);
  const auto& ct181 = cc->ModReduce(ct77);
  const auto& ct182 = cc->EvalMultNoRelin(ct181, ct181);
  const auto& ct183 = cc->Relinearize(ct182);
  const auto& ct184 = cc->ModReduce(ct79);
  const auto& ct185 = cc->EvalMultNoRelin(ct184, ct184);
  const auto& ct186 = cc->Relinearize(ct185);
  const auto& ct187 = cc->ModReduce(ct81);
  const auto& ct188 = cc->EvalMultNoRelin(ct187, ct187);
  const auto& ct189 = cc->Relinearize(ct188);
  const auto& ct190 = cc->ModReduce(ct83);
  const auto& ct191 = cc->EvalMultNoRelin(ct190, ct190);
  const auto& ct192 = cc->Relinearize(ct191);
  const auto& ct193 = cc->ModReduce(ct85);
  const auto& ct194 = cc->EvalMultNoRelin(ct193, ct193);
  const auto& ct195 = cc->Relinearize(ct194);
  const auto& ct196 = cc->ModReduce(ct87);
  const auto& ct197 = cc->EvalMultNoRelin(ct196, ct196);
  const auto& ct198 = cc->Relinearize(ct197);
  const auto& ct199 = cc->ModReduce(ct89);
  const auto& ct200 = cc->EvalMultNoRelin(ct199, ct199);
  const auto& ct201 = cc->Relinearize(ct200);
  const auto& ct202 = cc->ModReduce(ct91);
  const auto& ct203 = cc->EvalMultNoRelin(ct202, ct202);
  const auto& ct204 = cc->Relinearize(ct203);
  const auto& ct205 = cc->ModReduce(ct93);
  const auto& ct206 = cc->EvalMultNoRelin(ct205, ct205);
  const auto& ct207 = cc->Relinearize(ct206);
  const auto& ct208 = cc->ModReduce(ct95);
  const auto& ct209 = cc->EvalMultNoRelin(ct208, ct208);
  const auto& ct210 = cc->Relinearize(ct209);
  const auto& ct211 = cc->ModReduce(ct97);
  const auto& ct212 = cc->EvalMultNoRelin(ct211, ct211);
  const auto& ct213 = cc->Relinearize(ct212);
  const auto& ct214 = cc->ModReduce(ct99);
  const auto& ct215 = cc->EvalMultNoRelin(ct214, ct214);
  const auto& ct216 = cc->Relinearize(ct215);
  const auto& ct217 = cc->ModReduce(ct101);
  const auto& ct218 = cc->EvalMultNoRelin(ct217, ct217);
  const auto& ct219 = cc->Relinearize(ct218);
  const auto& ct220 = cc->ModReduce(ct103);
  const auto& ct221 = cc->EvalMultNoRelin(ct220, ct220);
  const auto& ct222 = cc->Relinearize(ct221);
  const auto& ct223 = cc->ModReduce(ct105);
  const auto& ct224 = cc->EvalMultNoRelin(ct223, ct223);
  const auto& ct225 = cc->Relinearize(ct224);
  const auto& ct226 = cc->ModReduce(ct107);
  const auto& ct227 = cc->EvalMultNoRelin(ct226, ct226);
  const auto& ct228 = cc->Relinearize(ct227);
  const auto& ct229 = cc->ModReduce(ct109);
  const auto& ct230 = cc->EvalMultNoRelin(ct229, ct229);
  const auto& ct231 = cc->Relinearize(ct230);
  const auto& ct232 = cc->ModReduce(ct111);
  const auto& ct233 = cc->EvalMultNoRelin(ct232, ct232);
  const auto& ct234 = cc->Relinearize(ct233);
  const auto& ct235 = cc->ModReduce(ct113);
  const auto& ct236 = cc->EvalMultNoRelin(ct235, ct235);
  const auto& ct237 = cc->Relinearize(ct236);
  const auto& ct238 = cc->ModReduce(ct115);
  const auto& ct239 = cc->EvalMultNoRelin(ct238, ct238);
  const auto& ct240 = cc->Relinearize(ct239);
  const auto& ct241 = cc->ModReduce(ct117);
  const auto& ct242 = cc->EvalMultNoRelin(ct241, ct241);
  const auto& ct243 = cc->Relinearize(ct242);
  const auto& ct244 = cc->ModReduce(ct119);
  const auto& ct245 = cc->EvalMultNoRelin(ct244, ct244);
  const auto& ct246 = cc->Relinearize(ct245);
  const auto& ct247 = cc->ModReduce(ct121);
  const auto& ct248 = cc->EvalMultNoRelin(ct247, ct247);
  const auto& ct249 = cc->Relinearize(ct248);
  const auto& ct250 = cc->ModReduce(ct123);
  const auto& ct251 = cc->EvalMultNoRelin(ct250, ct250);
  const auto& ct252 = cc->Relinearize(ct251);
  const auto& ct253 = cc->ModReduce(ct125);
  const auto& ct254 = cc->EvalMultNoRelin(ct253, ct253);
  const auto& ct255 = cc->Relinearize(ct254);
  const auto& ct256 = cc->ModReduce(ct127);
  const auto& ct257 = cc->EvalMultNoRelin(ct256, ct256);
  const auto& ct258 = cc->Relinearize(ct257);
  const auto& ct259 = cc->EvalAdd(ct162, ct165);
  const auto& ct260 = cc->EvalAdd(ct168, ct171);
  const auto& ct261 = cc->EvalAdd(ct259, ct260);
  const auto& ct262 = cc->EvalAdd(ct174, ct177);
  const auto& ct263 = cc->EvalAdd(ct180, ct183);
  const auto& ct264 = cc->EvalAdd(ct262, ct263);
  const auto& ct265 = cc->EvalAdd(ct261, ct264);
  const auto& ct266 = cc->EvalAdd(ct186, ct189);
  const auto& ct267 = cc->EvalAdd(ct192, ct195);
  const auto& ct268 = cc->EvalAdd(ct266, ct267);
  const auto& ct269 = cc->EvalAdd(ct198, ct201);
  const auto& ct270 = cc->EvalAdd(ct204, ct207);
  const auto& ct271 = cc->EvalAdd(ct269, ct270);
  const auto& ct272 = cc->EvalAdd(ct268, ct271);
  const auto& ct273 = cc->EvalAdd(ct265, ct272);
  const auto& ct274 = cc->EvalAdd(ct210, ct213);
  const auto& ct275 = cc->EvalAdd(ct216, ct219);
  const auto& ct276 = cc->EvalAdd(ct274, ct275);
  const auto& ct277 = cc->EvalAdd(ct222, ct225);
  const auto& ct278 = cc->EvalAdd(ct228, ct231);
  const auto& ct279 = cc->EvalAdd(ct277, ct278);
  const auto& ct280 = cc->EvalAdd(ct276, ct279);
  const auto& ct281 = cc->EvalAdd(ct234, ct237);
  const auto& ct282 = cc->EvalAdd(ct240, ct243);
  const auto& ct283 = cc->EvalAdd(ct281, ct282);
  const auto& ct284 = cc->EvalAdd(ct246, ct249);
  const auto& ct285 = cc->EvalAdd(ct252, ct255);
  const auto& ct286 = cc->EvalAdd(ct285, ct258);
  const auto& ct287 = cc->EvalAdd(ct284, ct286);
  const auto& ct288 = cc->EvalAdd(ct283, ct287);
  const auto& ct289 = cc->EvalAdd(ct280, ct288);
  const auto& ct290 = cc->EvalAdd(ct273, ct289);
  const auto& ct291 = cc->ModReduce(ct290);
  const auto& ct292 = cc->EvalMultNoRelin(ct291, ct291);
  const auto& ct293 = cc->Relinearize(ct292);
  const auto& ct294 = cc->ModReduce(ct165);
  const auto& ct295 = cc->EvalMultNoRelin(ct294, ct294);
  const auto& ct296 = cc->Relinearize(ct295);
  const auto& ct297 = cc->ModReduce(ct168);
  const auto& ct298 = cc->EvalMultNoRelin(ct297, ct297);
  const auto& ct299 = cc->Relinearize(ct298);
  const auto& ct300 = cc->ModReduce(ct171);
  const auto& ct301 = cc->EvalMultNoRelin(ct300, ct300);
  const auto& ct302 = cc->Relinearize(ct301);
  const auto& ct303 = cc->ModReduce(ct174);
  const auto& ct304 = cc->EvalMultNoRelin(ct303, ct303);
  const auto& ct305 = cc->Relinearize(ct304);
  const auto& ct306 = cc->ModReduce(ct177);
  const auto& ct307 = cc->EvalMultNoRelin(ct306, ct306);
  const auto& ct308 = cc->Relinearize(ct307);
  const auto& ct309 = cc->ModReduce(ct180);
  const auto& ct310 = cc->EvalMultNoRelin(ct309, ct309);
  const auto& ct311 = cc->Relinearize(ct310);
  const auto& ct312 = cc->ModReduce(ct183);
  const auto& ct313 = cc->EvalMultNoRelin(ct312, ct312);
  const auto& ct314 = cc->Relinearize(ct313);
  const auto& ct315 = cc->ModReduce(ct186);
  const auto& ct316 = cc->EvalMultNoRelin(ct315, ct315);
  const auto& ct317 = cc->Relinearize(ct316);
  const auto& ct318 = cc->ModReduce(ct189);
  const auto& ct319 = cc->EvalMultNoRelin(ct318, ct318);
  const auto& ct320 = cc->Relinearize(ct319);
  const auto& ct321 = cc->ModReduce(ct192);
  const auto& ct322 = cc->EvalMultNoRelin(ct321, ct321);
  const auto& ct323 = cc->Relinearize(ct322);
  const auto& ct324 = cc->ModReduce(ct195);
  const auto& ct325 = cc->EvalMultNoRelin(ct324, ct324);
  const auto& ct326 = cc->Relinearize(ct325);
  const auto& ct327 = cc->ModReduce(ct198);
  const auto& ct328 = cc->EvalMultNoRelin(ct327, ct327);
  const auto& ct329 = cc->Relinearize(ct328);
  const auto& ct330 = cc->ModReduce(ct201);
  const auto& ct331 = cc->EvalMultNoRelin(ct330, ct330);
  const auto& ct332 = cc->Relinearize(ct331);
  const auto& ct333 = cc->ModReduce(ct204);
  const auto& ct334 = cc->EvalMultNoRelin(ct333, ct333);
  const auto& ct335 = cc->Relinearize(ct334);
  const auto& ct336 = cc->ModReduce(ct207);
  const auto& ct337 = cc->EvalMultNoRelin(ct336, ct336);
  const auto& ct338 = cc->Relinearize(ct337);
  const auto& ct339 = cc->ModReduce(ct210);
  const auto& ct340 = cc->EvalMultNoRelin(ct339, ct339);
  const auto& ct341 = cc->Relinearize(ct340);
  const auto& ct342 = cc->ModReduce(ct213);
  const auto& ct343 = cc->EvalMultNoRelin(ct342, ct342);
  const auto& ct344 = cc->Relinearize(ct343);
  const auto& ct345 = cc->ModReduce(ct216);
  const auto& ct346 = cc->EvalMultNoRelin(ct345, ct345);
  const auto& ct347 = cc->Relinearize(ct346);
  const auto& ct348 = cc->ModReduce(ct219);
  const auto& ct349 = cc->EvalMultNoRelin(ct348, ct348);
  const auto& ct350 = cc->Relinearize(ct349);
  const auto& ct351 = cc->ModReduce(ct222);
  const auto& ct352 = cc->EvalMultNoRelin(ct351, ct351);
  const auto& ct353 = cc->Relinearize(ct352);
  const auto& ct354 = cc->ModReduce(ct225);
  const auto& ct355 = cc->EvalMultNoRelin(ct354, ct354);
  const auto& ct356 = cc->Relinearize(ct355);
  const auto& ct357 = cc->ModReduce(ct228);
  const auto& ct358 = cc->EvalMultNoRelin(ct357, ct357);
  const auto& ct359 = cc->Relinearize(ct358);
  const auto& ct360 = cc->ModReduce(ct231);
  const auto& ct361 = cc->EvalMultNoRelin(ct360, ct360);
  const auto& ct362 = cc->Relinearize(ct361);
  const auto& ct363 = cc->ModReduce(ct234);
  const auto& ct364 = cc->EvalMultNoRelin(ct363, ct363);
  const auto& ct365 = cc->Relinearize(ct364);
  const auto& ct366 = cc->ModReduce(ct237);
  const auto& ct367 = cc->EvalMultNoRelin(ct366, ct366);
  const auto& ct368 = cc->Relinearize(ct367);
  const auto& ct369 = cc->ModReduce(ct240);
  const auto& ct370 = cc->EvalMultNoRelin(ct369, ct369);
  const auto& ct371 = cc->Relinearize(ct370);
  const auto& ct372 = cc->ModReduce(ct243);
  const auto& ct373 = cc->EvalMultNoRelin(ct372, ct372);
  const auto& ct374 = cc->Relinearize(ct373);
  const auto& ct375 = cc->ModReduce(ct246);
  const auto& ct376 = cc->EvalMultNoRelin(ct375, ct375);
  const auto& ct377 = cc->Relinearize(ct376);
  const auto& ct378 = cc->ModReduce(ct249);
  const auto& ct379 = cc->EvalMultNoRelin(ct378, ct378);
  const auto& ct380 = cc->Relinearize(ct379);
  const auto& ct381 = cc->ModReduce(ct252);
  const auto& ct382 = cc->EvalMultNoRelin(ct381, ct381);
  const auto& ct383 = cc->Relinearize(ct382);
  const auto& ct384 = cc->ModReduce(ct255);
  const auto& ct385 = cc->EvalMultNoRelin(ct384, ct384);
  const auto& ct386 = cc->Relinearize(ct385);
  const auto& ct387 = cc->ModReduce(ct258);
  const auto& ct388 = cc->EvalMultNoRelin(ct387, ct387);
  const auto& ct389 = cc->Relinearize(ct388);
  const auto& ct390 = cc->EvalAdd(ct293, ct296);
  const auto& ct391 = cc->EvalAdd(ct299, ct302);
  const auto& ct392 = cc->EvalAdd(ct390, ct391);
  const auto& ct393 = cc->EvalAdd(ct305, ct308);
  const auto& ct394 = cc->EvalAdd(ct311, ct314);
  const auto& ct395 = cc->EvalAdd(ct393, ct394);
  const auto& ct396 = cc->EvalAdd(ct392, ct395);
  const auto& ct397 = cc->EvalAdd(ct317, ct320);
  const auto& ct398 = cc->EvalAdd(ct323, ct326);
  const auto& ct399 = cc->EvalAdd(ct397, ct398);
  const auto& ct400 = cc->EvalAdd(ct329, ct332);
  const auto& ct401 = cc->EvalAdd(ct335, ct338);
  const auto& ct402 = cc->EvalAdd(ct400, ct401);
  const auto& ct403 = cc->EvalAdd(ct399, ct402);
  const auto& ct404 = cc->EvalAdd(ct396, ct403);
  const auto& ct405 = cc->EvalAdd(ct341, ct344);
  const auto& ct406 = cc->EvalAdd(ct347, ct350);
  const auto& ct407 = cc->EvalAdd(ct405, ct406);
  const auto& ct408 = cc->EvalAdd(ct353, ct356);
  const auto& ct409 = cc->EvalAdd(ct359, ct362);
  const auto& ct410 = cc->EvalAdd(ct408, ct409);
  const auto& ct411 = cc->EvalAdd(ct407, ct410);
  const auto& ct412 = cc->EvalAdd(ct365, ct368);
  const auto& ct413 = cc->EvalAdd(ct371, ct374);
  const auto& ct414 = cc->EvalAdd(ct412, ct413);
  const auto& ct415 = cc->EvalAdd(ct377, ct380);
  const auto& ct416 = cc->EvalAdd(ct383, ct386);
  const auto& ct417 = cc->EvalAdd(ct416, ct389);
  const auto& ct418 = cc->EvalAdd(ct415, ct417);
  const auto& ct419 = cc->EvalAdd(ct414, ct418);
  const auto& ct420 = cc->EvalAdd(ct411, ct419);
  const auto& ct421 = cc->EvalAdd(ct404, ct420);
  const auto& ct422 = cc->ModReduce(ct421);
  const auto& ct423 = cc->EvalMultNoRelin(ct422, ct422);
  const auto& ct424 = cc->ModReduce(ct296);
  const auto& ct425 = cc->EvalMultNoRelin(ct424, ct424);
  const auto& ct426 = cc->ModReduce(ct299);
  const auto& ct427 = cc->EvalMultNoRelin(ct426, ct426);
  const auto& ct428 = cc->ModReduce(ct302);
  const auto& ct429 = cc->EvalMultNoRelin(ct428, ct428);
  const auto& ct430 = cc->ModReduce(ct305);
  const auto& ct431 = cc->EvalMultNoRelin(ct430, ct430);
  const auto& ct432 = cc->ModReduce(ct308);
  const auto& ct433 = cc->EvalMultNoRelin(ct432, ct432);
  const auto& ct434 = cc->ModReduce(ct311);
  const auto& ct435 = cc->EvalMultNoRelin(ct434, ct434);
  const auto& ct436 = cc->ModReduce(ct314);
  const auto& ct437 = cc->EvalMultNoRelin(ct436, ct436);
  const auto& ct438 = cc->ModReduce(ct317);
  const auto& ct439 = cc->EvalMultNoRelin(ct438, ct438);
  const auto& ct440 = cc->ModReduce(ct320);
  const auto& ct441 = cc->EvalMultNoRelin(ct440, ct440);
  const auto& ct442 = cc->ModReduce(ct323);
  const auto& ct443 = cc->EvalMultNoRelin(ct442, ct442);
  const auto& ct444 = cc->ModReduce(ct326);
  const auto& ct445 = cc->EvalMultNoRelin(ct444, ct444);
  const auto& ct446 = cc->ModReduce(ct329);
  const auto& ct447 = cc->EvalMultNoRelin(ct446, ct446);
  const auto& ct448 = cc->ModReduce(ct332);
  const auto& ct449 = cc->EvalMultNoRelin(ct448, ct448);
  const auto& ct450 = cc->ModReduce(ct335);
  const auto& ct451 = cc->EvalMultNoRelin(ct450, ct450);
  const auto& ct452 = cc->ModReduce(ct338);
  const auto& ct453 = cc->EvalMultNoRelin(ct452, ct452);
  const auto& ct454 = cc->ModReduce(ct341);
  const auto& ct455 = cc->EvalMultNoRelin(ct454, ct454);
  const auto& ct456 = cc->ModReduce(ct344);
  const auto& ct457 = cc->EvalMultNoRelin(ct456, ct456);
  const auto& ct458 = cc->ModReduce(ct347);
  const auto& ct459 = cc->EvalMultNoRelin(ct458, ct458);
  const auto& ct460 = cc->ModReduce(ct350);
  const auto& ct461 = cc->EvalMultNoRelin(ct460, ct460);
  const auto& ct462 = cc->ModReduce(ct353);
  const auto& ct463 = cc->EvalMultNoRelin(ct462, ct462);
  const auto& ct464 = cc->ModReduce(ct356);
  const auto& ct465 = cc->EvalMultNoRelin(ct464, ct464);
  const auto& ct466 = cc->ModReduce(ct359);
  const auto& ct467 = cc->EvalMultNoRelin(ct466, ct466);
  const auto& ct468 = cc->ModReduce(ct362);
  const auto& ct469 = cc->EvalMultNoRelin(ct468, ct468);
  const auto& ct470 = cc->ModReduce(ct365);
  const auto& ct471 = cc->EvalMultNoRelin(ct470, ct470);
  const auto& ct472 = cc->ModReduce(ct368);
  const auto& ct473 = cc->EvalMultNoRelin(ct472, ct472);
  const auto& ct474 = cc->ModReduce(ct371);
  const auto& ct475 = cc->EvalMultNoRelin(ct474, ct474);
  const auto& ct476 = cc->ModReduce(ct374);
  const auto& ct477 = cc->EvalMultNoRelin(ct476, ct476);
  const auto& ct478 = cc->ModReduce(ct377);
  const auto& ct479 = cc->EvalMultNoRelin(ct478, ct478);
  const auto& ct480 = cc->ModReduce(ct380);
  const auto& ct481 = cc->EvalMultNoRelin(ct480, ct480);
  const auto& ct482 = cc->ModReduce(ct383);
  const auto& ct483 = cc->EvalMultNoRelin(ct482, ct482);
  const auto& ct484 = cc->ModReduce(ct386);
  const auto& ct485 = cc->EvalMultNoRelin(ct484, ct484);
  const auto& ct486 = cc->ModReduce(ct389);
  const auto& ct487 = cc->EvalMultNoRelin(ct486, ct486);
  const auto& ct488 = cc->EvalAdd(ct423, ct425);
  const auto& ct489 = cc->EvalAdd(ct427, ct429);
  const auto& ct490 = cc->EvalAdd(ct488, ct489);
  const auto& ct491 = cc->EvalAdd(ct431, ct433);
  const auto& ct492 = cc->EvalAdd(ct435, ct437);
  const auto& ct493 = cc->EvalAdd(ct491, ct492);
  const auto& ct494 = cc->EvalAdd(ct490, ct493);
  const auto& ct495 = cc->EvalAdd(ct439, ct441);
  const auto& ct496 = cc->EvalAdd(ct443, ct445);
  const auto& ct497 = cc->EvalAdd(ct495, ct496);
  const auto& ct498 = cc->EvalAdd(ct447, ct449);
  const auto& ct499 = cc->EvalAdd(ct451, ct453);
  const auto& ct500 = cc->EvalAdd(ct498, ct499);
  const auto& ct501 = cc->EvalAdd(ct497, ct500);
  const auto& ct502 = cc->EvalAdd(ct494, ct501);
  const auto& ct503 = cc->EvalAdd(ct455, ct457);
  const auto& ct504 = cc->EvalAdd(ct459, ct461);
  const auto& ct505 = cc->EvalAdd(ct503, ct504);
  const auto& ct506 = cc->EvalAdd(ct463, ct465);
  const auto& ct507 = cc->EvalAdd(ct467, ct469);
  const auto& ct508 = cc->EvalAdd(ct506, ct507);
  const auto& ct509 = cc->EvalAdd(ct505, ct508);
  const auto& ct510 = cc->EvalAdd(ct471, ct473);
  const auto& ct511 = cc->EvalAdd(ct475, ct477);
  const auto& ct512 = cc->EvalAdd(ct510, ct511);
  const auto& ct513 = cc->EvalAdd(ct479, ct481);
  const auto& ct514 = cc->EvalAdd(ct483, ct485);
  const auto& ct515 = cc->EvalAdd(ct514, ct487);
  const auto& ct516 = cc->EvalAdd(ct513, ct515);
  const auto& ct517 = cc->EvalAdd(ct512, ct516);
  const auto& ct518 = cc->EvalAdd(ct509, ct517);
  const auto& ct519 = cc->EvalAdd(ct502, ct518);
  const auto& ct520 = cc->Relinearize(ct519);
  const auto& ct521 = cc->ModReduce(ct520);
  return ct521;
}
CiphertextT func__encrypt__arg0(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg1(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg2(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg3(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg4(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg5(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg6(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg7(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg8(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg9(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg10(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg11(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg12(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg13(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg14(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg15(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg16(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg17(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg18(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg19(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg20(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg21(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg22(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg23(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg24(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg25(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg26(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg27(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg28(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg29(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg30(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg31(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg32(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg33(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg34(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg35(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg36(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg37(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg38(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg39(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg40(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg41(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg42(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg43(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg44(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg45(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg46(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg47(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg48(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg49(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg50(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg51(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg52(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg53(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg54(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg55(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg56(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg57(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg58(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg59(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg60(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg61(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg62(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
CiphertextT func__encrypt__arg63(CryptoContextT cc, std::vector<int16_t> v0, PublicKeyT pk) {
  std::vector<int64_t> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakePackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
std::vector<int16_t> func__decrypt__result0(CryptoContextT cc, CiphertextT ct, PrivateKeyT sk) {
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(8);
  const auto& v0_cast = pt->GetPackedValue();
  std::vector<int16_t> v0(std::begin(v0_cast), std::end(v0_cast));
  return v0;
}
CryptoContextT func__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(4);
  params.SetPlaintextModulus(65537);
  params.SetFirstModSize(35);
  params.SetScalingModSize(44);
  params.SetScalingTechnique(FIXEDMANUAL);
  params.SetRingDim(16384);
  params.SetSecurityLevel(lbcrypto::HEStd_NotSet);
  params.SetKeySwitchTechnique(HYBRID);
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
