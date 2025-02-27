#include "lib/Analysis/NoiseAnalysis/BGV/NoiseByMonoBoundCoeffModel.h"

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

// the formulae below are mainly taken from MMLGA22
// "Finding and Evaluating Parameters for BGV"
// https://eprint.iacr.org/2022/706

template <bool P>
using Model = NoiseByMonoBoundCoeffModel<P>;

template <bool P>
double Model<P>::toLogBound(const LocalParamType &param,
                               const StateType &noise) {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  // StateType only stores e in (m + t * e), so when we want to print the bound
  // we need to multiply t back
  return log(t * noise.getValue()) / log(2);
}

template <bool P>
double Model<P>::toLogBudget(const LocalParamType &param,
                                const StateType &noise) {
  return toLogTotal(param) - toLogBound(param, noise);
}

template <bool P>
double Model<P>::toLogTotal(const LocalParamType &param) {
  double total = 0;
  auto logqi = param.getSchemeParam()->getLogqi();
  for (auto i = 0; i <= param.getCurrentLevel(); ++i) {
    total += logqi[i];
  }
  return total - 1.0;
}

template <bool P>
std::string Model<P>::toLogBoundString(const LocalParamType &param,
                                          const StateType &noise) {
  auto logBound = toLogBound(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBound;
  return stream.str();
}

template <bool P>
std::string Model<P>::toLogBudgetString(const LocalParamType &param,
                                           const StateType &noise) {
  auto logBudget = toLogBudget(param, noise);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logBudget;
  return stream.str();
}

template <bool P>
std::string Model<P>::toLogTotalString(const LocalParamType &param) {
  auto logTotal = toLogTotal(param);
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << logTotal;
  return stream.str();
}

template<bool P>
double Model<P>::getExpansionFactor(const LocalParamType &param) {
  auto phi = param.getSchemeParam()->getRingDim() / 2;
  return phi;  
}

template<bool P>
double Model<P>::getBoundErr(const LocalParamType &param) {
  auto std0 = param.getSchemeParam()->getStd0();
  // probability of larger than 6 * std0 is less than 2^{-30}
  auto assurance = 6;
  auto boundErr = std0 * assurance;
  return boundErr;
}

template<bool P>
double Model<P>::getBoundKey(const LocalParamType &param) {
  // assume UNIFORM_TERNARY
  auto boundKey = 1.0;
  return boundKey;
}

template <bool P>
double Model<P>::getVarianceErr(const LocalParamType &param) {
  auto std0 = param.getSchemeParam()->getStd0();
  return std0 * std0;
}

template <bool P>
double Model<P>::getVarianceKey(const LocalParamType &param) {
  // assume UNIFORM_TERNARY
  return 2.0 / 3.0;
}

template <bool P>
double Model<P>::getFactorD(const LocalParamType &param) {
  // probability that a exceeds its standard deviation by more than a factor of D
  // is roughly erfc(D) with erfc(6) = 2^-55, erfc(5) = 2^-40, erfc(4.5) = 2^-32
  return 6.;
}

template<bool P>
double Model<P>::getBScale(const LocalParamType &param) {
  auto varianceKey = getVarianceKey(param);
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto d = getFactorD(param);
  auto phi = getPhi(param);

  // B_scale = D * t * sqrt(phi(m)/12 * (1 + phi(m) * V_key)
  double innerTerm = (phi / 12.) * (1 + phi * varianceKey);
  return d * t * sqrt(innerTerm);  
}

template<bool P>
double Model<P>::getPhi(const LocalParamType &param) {
  return param.getSchemeParam()->getRingDim() / 2.;  
}

template<bool P>
typename Model<P>::StateType Model<P>::evalEncryptPk(
    const LocalParamType &param) {
      
  auto varianceError = getVarianceErr(param);
  // uniform ternary
  auto varianceKey = getVarianceKey(param);
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto d = getFactorD(param);
  auto phi = getPhi(param);
    
  // public key (-as + t * e, a)
  // public key encryption (-aus + t(u * e + e_0) + m, au + e_1)
  // ||m + t * (u * e + e_1 * s + e_0)||
  // <= D * t * sqrt(phi(m) * (1/12 + 2 * phi(m) * V_err * V_key + V_err))
  double innerTerm = phi * (1./12. + 2. * phi * varianceError * varianceKey + varianceKey);
  double fresh = d * t * sqrt(innerTerm);
  return StateType::of(fresh);
}

template<bool P>
typename Model<P>::StateType Model<P>::evalEncryptSk(
    const LocalParamType &param) {
  
  auto varianceError = getVarianceErr(param);
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto d = getFactorD(param);
  auto phi = getPhi(param);
    
  // secret key s
  // secret key encryption (-as + m + t * e, a)
  // ||m + t * (u * e + e_1 * s + e_0)||
  // <= D * t * sqrt(phi(m) * (1/12 + V_err))
  double innerTerm = phi * (1./12. + varianceError);
  double fresh = d * t * sqrt(innerTerm);
  return StateType::of(fresh);
}

template<bool P>
typename Model<P>::StateType Model<P>::evalEncrypt(
    const LocalParamType &param) {
  // P stands for public key encryption
  if constexpr (P) {
    return evalEncryptPk(param);
  } else {
    return evalEncryptSk(param);
  }
}

template<bool P>
typename Model<P>::StateType Model<P>::evalConstant(
    const LocalParamType &param) {
  auto t = param.getSchemeParam()->getPlaintextModulus();
  auto phi = getPhi(param);
  
  // noise part of the plaintext in a pt-ct multiplication
  // v_const <= t * sqrt(phi(m) / 12)
  return StateType::of(t * sqrt(phi / 12.0));
}

template<bool P>
typename Model<P>::StateType Model<P>::evalAdd(const StateType &lhs,
                                                     const StateType &rhs) {
  // v_add <= v_0 + v_1
  return StateType::of(lhs.getValue() + rhs.getValue());
}

template<bool P>
typename Model<P>::StateType Model<P>::evalMul(
    const LocalParamType &resultParam, const StateType &lhs,
    const StateType &rhs) {
  // v_mul <= v_0 * v_1
  return StateType::of(lhs.getValue() * rhs.getValue());
}

template<bool P>
typename Model<P>::StateType Model<P>::evalModReduce(
    const LocalParamType &inputParam, const StateType &input) {

  auto currentLogqi =
      inputParam.getSchemeParam()->getLogqi()[inputParam.getCurrentLevel()];
  double modulus = pow(2.0, currentLogqi);

  // modulus switching is essentially a scaling operation
  // so the original error is scaled by the modulus
  // ||v_scaled|| = ||v_input|| / modulus
  auto scaled = input.getValue() / modulus;
  // in the meantime, it will introduce a rounding error
  // (tau_0, tau_1) to (ct_0, ct_1) 
  // ||tau_0 + tau_1 * s|| <= D * t * sqrt(phi(m)/12 * (1 + phi(m) * V_key) = B_scale
  // ||v_ms|| <= ||v_scaled|| + B_scale
  double bScale = getBScale(inputParam);
  return StateType::of(scaled + bScale);
}

template<bool P>
typename Model<P>::StateType Model<P>::evalRelinearizeHYBRID(
    const LocalParamType &inputParam, const StateType &input) {
  // for v_input, after modup and moddown, it remains the same (with rounding).
  // We only need to consider the error from key switching key
  // and rounding error during moddown.
  // Check section 3.2 of MMLGA22 for more details.


  //TODO: implement mono keyswitch
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

template<bool P>
typename Model<P>::StateType Model<P>::evalRelinearize(
    const LocalParamType &inputParam, const StateType &input) {
  // assume HYBRID
  // if we further introduce BV to SchemeParam we can have alternative
  // implementation.
  return evalRelinearizeHYBRID(inputParam, input);
}

// instantiate template class
template class NoiseByMonoBoundCoeffModel<false>;
template class NoiseByMonoBoundCoeffModel<true>;

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
