#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByBoundCoeffModel.h"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <ios>
#include <numeric>
#include <sstream>
#include <string>

namespace mlir {
namespace heir {
namespace bgv {

// the formulae below are mainly taken from KPZ21
// "Revisiting Homomorphic Encryption Schemes for Finite Fields"
// https://ia.cr/2021/204

template <bool W, bool P>
using Model = NoiseByBoundCoeffModel<W, P>;

template <bool W, bool P>
double Model<W, P>::toLogBound(const LocalParamType &param,
                               const StateType &noise) {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  // StateType only stores e in (m + t * e), so when we want to print the bound
  // we need to multiply t back
  return log(t * noise.getValue()) / log(2);
}

template <bool W, bool P>
double Model<W, P>::toLogBudget(const LocalParamType &param,
                                const StateType &noise) {
  return toLogTotal(param) - toLogBound(param, noise);
}

template <bool W, bool P>
double Model<W, P>::toLogTotal(const LocalParamType &param) {
  double total = 0;
  auto logqi = param.getSchemeParam()->getLogqi();
  for (auto i = 0; i <= param.getCurrentLevel(); ++i) {
    total += logqi[i];
  }
  return total - 1.0;
}

template <bool W, bool P>
std::string Model<W, P>::toLogBoundString(const LocalParamType &param,
                                          const StateType &noise) {
  auto logBound = toLogBound(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBound;
  return stream.str();
}

template <bool W, bool P>
std::string Model<W, P>::toLogBudgetString(const LocalParamType &param,
                                           const StateType &noise) {
  auto logBudget = toLogBudget(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBudget;
  return stream.str();
}

template <bool W, bool P>
std::string Model<W, P>::toLogTotalString(const LocalParamType &param) {
  auto logTotal = toLogTotal(param);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logTotal;
  return stream.str();
}

template <bool W, bool P>
double Model<W, P>::getExpansionFactor(const LocalParamType &param) {
  auto n = param.getSchemeParam()->getRingDim();
  if constexpr (W) {
    // worst-case
    // well known from DPSZ12
    return n;
  } else {
    // average-case
    // experimental result
    // cite HPS19 and KPZ21
    return 2.0 * sqrt(n);
  }
}

template <bool W, bool P>
double Model<W, P>::getBoundErr(const LocalParamType &param) {
  auto std0 = param.getSchemeParam()->getStd0();
  // probability of larger than 6 * std0 is less than 2^{-30}
  auto assurance = 6;
  auto boundErr = std0 * assurance;
  return boundErr;
}

template <bool W, bool P>
double Model<W, P>::getBoundKey(const LocalParamType &param) {
  // assume UNIFORM_TERNARY
  auto boundKey = 1.0;
  return boundKey;
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalEncryptPk(
    const LocalParamType &param) {
  auto boundErr = getBoundErr(param);
  auto boundKey = getBoundKey(param);
  auto expansionFactor = getExpansionFactor(param);

  // public key (-as + t * e, a)
  // public key encryption (-aus + t(u * e + e_0) + m, au + e_1)
  // m + t * (u * e + e_1 * s + e_0)
  // v_fresh = u * e + e_1 * s + e_0
  double fresh = boundErr * (1. + 2. * expansionFactor * boundKey);
  return StateType::of(fresh);
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalEncryptSk(
    const LocalParamType &param) {
  auto boundErr = getBoundErr(param);

  // secret key s
  // secret key encryption (-as + m + t * e, a)
  // v_fresh = e
  double fresh = boundErr;
  return StateType::of(fresh);
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalEncrypt(
    const LocalParamType &param) {
  // P stands for public key encryption
  if constexpr (P) {
    return evalEncryptPk(param);
  } else {
    return evalEncryptSk(param);
  }
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalConstant(
    const LocalParamType &param) {
  // constant is m + t * 0
  // v_constant = 0
  return StateType::of(0);
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalAdd(const StateType &lhs,
                                                     const StateType &rhs) {
  // m_0 + tv_0 + m_1 + tv_1 <= [m_0 + m_1]_t + t(v_0 + v_1 + u)
  // v_add = v_0 + v_1 + u
  // where ||u|| <= 1
  return StateType::of(lhs.getValue() + rhs.getValue() + 1);
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalMul(
    const LocalParamType &resultParam, const StateType &lhs,
    const StateType &rhs) {
  auto t = resultParam.getSchemeParam()->getPlaintextModulus();
  auto expansionFactor = getExpansionFactor(resultParam);

  // (m_0 + tv_0) * (m_1 + tv_1) <=
  //   [m_0 * m_1]_t + t(v_0 * m_1 + v_1 * m_0 + v_0 * v_1 + r_m)
  // where m_0 * m_1 = [m_0 * m_1]_t + tr_m
  // ||r_m|| <= delta * t / 2, delta is the expansion factor
  // v_mul = v_0 * m_1 + v_1 * m_0 + v_0 * v_1 + r_m
  // ||v_mul|| <=
  //   (delta * t / 2) * (2 * ||v_0|| * ||v_1|| + ||v_0|| + ||v_1|| + 1)
  return StateType::of((expansionFactor * t / 2) *
                       (lhs.getValue() * rhs.getValue() * 2 + lhs.getValue() +
                        rhs.getValue() + 1));
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalModReduce(
    const LocalParamType &inputParam, const StateType &input) {
  // for cv > 2 the rounding error term is different!
  // like (tau_0, tau_1, tau_2) and the error becomes
  // tau_0 + tau_1 s + tau_2 s^2
  assert(inputParam.getDimension() == 2);

  auto currentLogqi =
      inputParam.getSchemeParam()->getLogqi()[inputParam.getCurrentLevel()];

  double modulus = pow(2.0, currentLogqi);

  auto expansionFactor = getExpansionFactor(inputParam);
  auto boundKey = getBoundKey(inputParam);

  // modulus switching is essentially a scaling operation
  // so the original error is scaled by the modulus
  // ||v_scaled|| = ||v_input|| / modulus
  auto scaled = input.getValue() / modulus;
  // in the meantime, it will introduce an rounding error
  // (tau_0, tau_1) to the (ct_0, ct_1) where ||tau_i|| < t / 2
  // so ||tau_0 + tau_1 * s|| <= t / 2 (1 + delta ||s||)
  // ||v_added|| <= (1 + delta * Bkey) / 2
  auto added = (1.0 + expansionFactor * boundKey) / 2;
  return StateType::of(scaled + added);
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalRelinearizeHYBRID(
    const LocalParamType &inputParam, const StateType &input) {
  // for v_input, after modup and moddown, it remains the same (with rounding).
  // We only need to consider the error from key switching key
  // and rounding error during moddown.
  // Check the B.1.3 section of KPZ21 for more details.

  // also note that for cv > 3 (e.g. multiplication), we need to relinearize
  // more terms like ct_3 and ct_4.
  // this is a common path for mult relinearize and rotation relinearize
  // so no assertion here for now.

  auto dnum = inputParam.getSchemeParam()->getDnum();
  auto expansionFactor = getExpansionFactor(inputParam);
  auto boundErr = getBoundErr(inputParam);
  auto boundKey = getBoundKey(inputParam);

  auto currentLevel = inputParam.getCurrentLevel();
  // modup from Ql to QlP, so one more digit
  auto currentNumDigit = ceil(static_cast<double>(currentLevel + 1) / dnum) + 1;

  // log(qiq_{i+1}...), the digit size for a certain digit
  // we use log(pip_{i+1}...) as an approximation,
  // as we often choose P > each digit
  auto logqi = inputParam.getSchemeParam()->getLogqi();
  double logDigitSize = std::accumulate(logqi.begin(), logqi.end(), 0.0);
  // omega in literature
  auto digitSize = pow(2.0, logDigitSize);

  // the HYBRID key switching error is
  // sum over all digit (ct_2 * e_ksk)
  // there are "currentNumDigit" digits
  // and ||c_2|| <= digitSize / 2
  // ||c_2 * e_ksk|| <= delta * digitSize * Berr / 2
  auto boundKeySwitch =
      currentNumDigit * digitSize * expansionFactor * boundErr / 2.0;

  // moddown by P
  auto scaled = boundKeySwitch / digitSize;

  // moddown added noise, similar to modreduce above.
  auto added = (1.0 + expansionFactor * boundKey) / 2;

  // for relinearization after multiplication, often scaled + added is far less
  // than input.
  return StateType::of(input.getValue() + scaled + added);
}

template <bool W, bool P>
typename Model<W, P>::StateType Model<W, P>::evalRelinearize(
    const LocalParamType &inputParam, const StateType &input) {
  // assume HYBRID
  // if we further introduce BV to SchemeParam we can have alternative
  // implementation.
  return evalRelinearizeHYBRID(inputParam, input);
}

// instantiate template class
template class NoiseByBoundCoeffModel<false, true>;
template class NoiseByBoundCoeffModel<true, true>;
template class NoiseByBoundCoeffModel<false, false>;
template class NoiseByBoundCoeffModel<true, false>;

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
