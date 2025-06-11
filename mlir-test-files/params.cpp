#include "params.h"
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

using namespace lbcrypto;

// Core function to generate JSON string from parameters
std::string getParamsAsJsonString(const CryptoContext<DCRTPoly>& cc, 
                                 const std::string& testname,
                                 const std::string& selectionApproach) {
  auto modulusChain = cc->GetCryptoParameters()->GetElementParams()->GetParams();
  uint32_t totalSize = cc->GetCryptoParameters()->GetElementParams()->GetModulus().GetMSB();
  uint32_t ringDimension = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension();
  uint32_t plaintextModulus = cc->GetCryptoParameters()->GetPlaintextModulus();
  
  // Get hybrid keyswitching parameters if available
  uint32_t extensionModuliSize = 0;
  uint32_t numPartQ = 0;
  uint32_t numPerPartQ = 0;
  std::vector<uint32_t> extensionModuliSizes; // Individual extension moduli sizes
  
  // Extract keyswitching parameters
  auto cryptoParams = cc->GetCryptoParameters();
  
  // Cast to access internal parameters
  auto parametersRNS = std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoParams);
  if (parametersRNS) {
    if (parametersRNS->GetKeySwitchTechnique() == HYBRID) {
      numPartQ = parametersRNS->GetNumPartQ();
      numPerPartQ = parametersRNS->GetNumPerPartQ();
      
      // Get extension moduli size
      auto paramsP = parametersRNS->GetParamsP();
      if (paramsP) {
        auto pModuliParams = paramsP->GetParams();
        extensionModuliSize = paramsP->GetModulus().GetMSB();
        
        // Extract individual extension moduli sizes
        for (auto & pModuliParam : pModuliParams) {
          extensionModuliSizes.push_back(pModuliParam->GetModulus().GetMSB());
        }
      }
    }
  }
  
  // Generate Unix timestamp in seconds
  auto now = std::chrono::system_clock::now();
  auto unix_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
      now.time_since_epoch()).count();
  
  std::stringstream ss;
  ss << "{\n";
  ss << R"(  "testname": ")" << testname << "\",\n";
  ss << R"(  "selectionApproach": ")" << selectionApproach << "\",\n";
  ss << R"(  "timestamp": )" << unix_timestamp << ",\n";
  ss << "  \"modulusSizes\": [";
  for (size_t i = 0; i < modulusChain.size(); ++i) {
    ss << modulusChain[i]->GetModulus().GetMSB();
    if (i != modulusChain.size() - 1) {
      ss << ", ";
    }
  }
  ss << "],\n";
  ss << "  \"totalSize\": " << totalSize << ",\n";
  ss << "  \"ringDimension\": " << ringDimension << ",\n";
  ss << "  \"plaintextModulus\": " << plaintextModulus << ",\n";
  ss << "  \"extensionModuliSize\": " << extensionModuliSize << ",\n";
  ss << "  \"extensionModuliSizes\": [";
  for (size_t i = 0; i < extensionModuliSizes.size(); ++i) {
    ss << extensionModuliSizes[i];
    if (i != extensionModuliSizes.size() - 1) {
      ss << ", ";
    }
  }
  ss << "],\n";
  ss << "  \"numPartQ\": " << numPartQ << ",\n";
  ss << "  \"numPerPartQ\": " << numPerPartQ << "\n";
  ss << "}";
  
  return ss.str();
}

// Print parameters to standard output
void printModulusChain(const CryptoContext<DCRTPoly>& cc, const std::string& testname,const std::string& selectionApproach) {
  std::cout << getParamsAsJsonString(cc, testname, selectionApproach) << std::endl;
}

// Save parameters to a file
bool saveParamsToJsonFile(const CryptoContext<DCRTPoly>& cc, 
                         const std::string& testname, 
                         const std::string& selectionApproach) {
    // Create filename with timestamp
    std::filesystem::path cwd = std::filesystem::current_path();
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    std::string filename = (cwd / "data" / (testname + "_" + selectionApproach + "_execution_" + timestamp.str() + ".json")).string();
    
    // Check if the directory exists, create if it doesn't
    std::filesystem::path dir = "data/";
    if (!std::filesystem::exists(dir)) {
        if (!std::filesystem::create_directory(dir)) {
            std::cerr << "Failed to create directory: " << dir << std::endl;
            return false;
        }
    }
    
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    std::string jsonString = getParamsAsJsonString(cc, testname, selectionApproach);
    if (jsonString.empty()) {
        std::cerr << "Failed to generate JSON string" << std::endl;
        return false;
    }
    
    outFile << jsonString << std::endl;
    
    if (outFile.fail()) {
        std::cerr << "Failed to write to file: " << filename << std::endl;
        return false;
    }
    
    outFile.close();
    if (outFile.fail()) {
        std::cerr << "Failed to close file: " << filename << std::endl;
        return false;
    }
    
    std::cerr << "Successfully wrote to file: " << std::filesystem::absolute(filename) << std::endl;
    return true;
}



void EvalNoiseBGV(std::string tag, CryptoContext<DCRTPoly> cryptoContext,
                                  PrivateKey<DCRTPoly> privateKey,
                                  ConstCiphertext<DCRTPoly> ciphertext) {
    Plaintext ptxt;
    cryptoContext->Decrypt(privateKey, ciphertext, &ptxt);
    ptxt->SetLength(8);

    const std::vector<DCRTPoly>& cv = ciphertext->GetElements();
    DCRTPoly s = privateKey->GetPrivateElement();

    size_t sizeQl = cv[0].GetParams()->GetParams().size();
    size_t sizeQs = s.GetParams()->GetParams().size();
    size_t diffQl = sizeQs - sizeQl;

    auto scopy(s);
    scopy.DropLastElements(diffQl);

    DCRTPoly sPower(scopy);
    DCRTPoly b = cv[0];
    b.SetFormat(Format::EVALUATION);

    DCRTPoly ci;
    for (size_t i = 1; i < cv.size(); i++) {
        ci = cv[i];
        ci.SetFormat(Format::EVALUATION);

        b += sPower * ci;
        sPower *= scopy;
    }

    b.SetFormat(Format::COEFFICIENT);
    Poly b_big = b.CRTInterpolate();

    Poly plain_big;

    DCRTPoly plain_dcrt = ptxt->GetElement<DCRTPoly>();
    auto plain_dcrt_size = plain_dcrt.GetNumOfElements();

    if (plain_dcrt_size > 0) {
        plain_dcrt.SetFormat(Format::COEFFICIENT);
        plain_big = plain_dcrt.CRTInterpolate();
    } else {
        std::vector<int64_t> value = ptxt->GetPackedValue();
        Plaintext repack = cryptoContext->MakePackedPlaintext(value);
        DCRTPoly plain_repack = repack->GetElement<DCRTPoly>();
        plain_repack.SetFormat(Format::COEFFICIENT);
        plain_big = plain_repack.CRTInterpolate();
    }

    auto plain_modulus = plain_big.GetModulus();
    auto b_modulus = b_big.GetModulus();
    plain_big.SwitchModulus(b_big.GetModulus(), b_big.GetRootOfUnity(), 0, 0);

    Poly res = b_big - plain_big;

    double noise = (log2(res.Norm()));
    double logQ = 0;
    std::vector<double> logqi_v;
    for (usint i = 0; i < sizeQl; i++) {
        double logqi =
            log2(cv[0].GetParams()->GetParams()[i]->GetModulus().ConvertToInt());
        logqi_v.push_back(logqi);
        logQ += logqi;
    }

    std::cout << tag << "\t\t" << "cv " << cv.size() << " Ql " << sizeQl
              << " budget " << logQ - noise - 1 << " noise: " << noise << std::endl;
}
