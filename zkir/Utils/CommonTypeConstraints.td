#ifndef ZKIR_UTILS_COMMONTYPECONSTRAINTS_TD_
#define ZKIR_UTILS_COMMONTYPECONSTRAINTS_TD_

include "mlir/IR/CommonTypeConstraints.td"

class Static1DTensorOf<list<Type> allowedTypes>
  : RankedTensorOf<allowedTypes, [HasAnyRankOfPred<[1]>, HasStaticShapePred],
                   "statically shaped 1D tensor">;

#endif  // ZKIR_UTILS_COMMONTYPECONSTRAINTS_TD_
