// Pattern suffix list for canonicalization patterns.
// Usage:
//   #include "zkir/Utils/CanonicalizationPatterns.inc"
//   #define ZKIR_ADD_PATTERN(Name) patterns.add<Prefix##Name>(context);
//   ZKIR_ADD_PATTERN_LIST(ZKIR_ADD_PATTERN)
//   #undef ZKIR_ADD_PATTERN

// AddOp patterns.
#define ZKIR_ADD_PATTERN_LIST(X) \
  X(AddConstantTwice)            \
  X(AddConstantToSubLhs)         \
  X(AddConstantToSubRhs)         \
  X(AddSelfIsDouble)             \
  X(AddBothNegated)              \
  X(AddAfterSub)                 \
  X(AddAfterNegLhs)              \
  X(AddAfterNegRhs)              \
  X(FactorMulAdd)

// SubOp patterns.
#define ZKIR_SUB_PATTERN_LIST(X) \
  X(SubConstantFromAdd)          \
  X(SubConstantTwiceLhs)         \
  X(SubConstantTwiceRhs)         \
  X(SubAddFromConstant)          \
  X(SubSubFromConstantLhs)       \
  X(SubSubFromConstantRhs)       \
  X(SubSelfIsZero)               \
  X(SubLhsAfterAdd)              \
  X(SubRhsAfterAdd)              \
  X(SubLhsAfterSub)              \
  X(SubAfterNegLhs)              \
  X(SubAfterNegRhs)              \
  X(SubBothNegated)              \
  X(SubAfterSquareBoth)          \
  X(SubAfterSumSquare)           \
  X(FactorMulSub)

// MulOp patterns.
#define ZKIR_MUL_PATTERN_LIST(X) \
  X(MulByTwoIsDouble)            \
  X(MulSelfIsSquare)             \
  X(MulNegativeOneRhs)           \
  X(MulNegativeTwoRhs)           \
  X(MulNegativeThreeRhs)         \
  X(MulNegativeFourRhs)          \
  X(MulConstantTwice)            \
  X(MulOfMulByConstant)          \
  X(MulAddDistributeConstant)    \
  X(MulSubDistributeConstantRhs) \
  X(MulSubDistributeConstantLhs)
