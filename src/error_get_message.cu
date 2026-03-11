#include "ra/error.cuh"

namespace ra {

void
Error::get_message(std::string& output) const {
  switch (category) {
  case (ErrorCategory::Host):
    output = "Host error: ";
    break;
  case (ErrorCategory::Device):
    output = "Device error: ";
    break;
  default:
    output = "Unknown error: ";
  }

  output += std::to_string(value);
}

} // namespace ra
