#pragma once

#include <string>
#include "src/pke/include/openfhe.h" 

using namespace lbcrypto;

/**
 * Converts cryptographic parameters to a JSON string
 * 
 * @param cc The cryptographic context
 * @param testname Name of the test
 * @param selectionApproach Approach used for parameter selection
 * @return A JSON string containing the parameters
 */
std::string getParamsAsJsonString(const CryptoContext<DCRTPoly>& cc, 
                                 const std::string& testname,
                                 const std::string& selectionApproach);

/**
 * Prints the modulus chain to standard output (legacy version)
 * 
 * @param cc The cryptographic context
 */
void printModulusChain(const CryptoContext<DCRTPoly>& cc);

/**
 * Prints the modulus chain to standard output
 *
 * @param cc The cryptographic context
 * @param testname Name of the test
 * @param selectionApproach Approach used for parameter selection
 */
void printModulusChain(const CryptoContext<DCRTPoly>& cc,
                       const std::string& testname,
                       const std::string& selectionApproach);

/**
 * Saves parameters to a JSON file
 * 
 * @param cc The cryptographic context
 * @param testname Name of the test
 * @param selectionApproach Approach used for parameter selection
 * @return True if file was successfully written, false otherwise
 */
bool saveParamsToJsonFile(const CryptoContext<DCRTPoly>& cc, 
                         const std::string& testname, 
                         const std::string& selectionApproach);