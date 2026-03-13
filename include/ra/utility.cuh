#pragma once

#include <cuda/std/type_traits>

namespace ra {

template <typename T>
constexpr typename cuda::std::underlying_type<T>::type
to_underlying(T input) noexcept {
  return static_cast<typename cuda::std::underlying_type<T>::type>(input);
}

} // namespace ra
