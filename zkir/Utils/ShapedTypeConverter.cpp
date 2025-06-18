#include "zkir/Utils/ShapedTypeConverter.h"

#include <assert.h>

namespace mlir::zkir {

// static
Type ShapedTypeConverter::convertShapedType(ShapedType oldType,
                                            ArrayRef<int64_t> shape,
                                            Type elementType) {
  if (auto memrefType = dyn_cast<MemRefType>(oldType)) {
    return MemRefType::get(shape, elementType);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(oldType)) {
    return tensorType.cloneWith(shape, elementType);
  }
  assert(false && "Unsupported shaped type");
  return oldType;
}

}  // namespace mlir::zkir
