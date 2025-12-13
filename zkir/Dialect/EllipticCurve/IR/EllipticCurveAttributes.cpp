/* Copyright 2025 The ZKIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"

#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::elliptic_curve {

// static
Attribute ShortWeierstrassAttr::parse(AsmParser &parser, Type odsType) {
  Attribute a, b, gX, gY;
  Type baseField;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(a)) ||
      failed(parser.parseComma()) || failed(parser.parseAttribute(b)) ||
      failed(parser.parseComma()) || failed(parser.parseLParen()) ||
      failed(parser.parseAttribute(gX)) || failed(parser.parseComma()) ||
      failed(parser.parseAttribute(gY)) || failed(parser.parseRParen()) ||
      failed(parser.parseGreater()) ||
      failed(field::parseColonFieldType(parser, baseField)))
    return nullptr;

  if (failed(field::validateAttribute(parser, baseField, a, "a")) ||
      failed(field::validateAttribute(parser, baseField, b, "b")) ||
      failed(field::validateAttribute(parser, baseField, gX, "gX")) ||
      failed(field::validateAttribute(parser, baseField, gY, "gY")))
    return nullptr;

  a = field::maybeToMontgomery(baseField, a);
  b = field::maybeToMontgomery(baseField, b);
  gX = field::maybeToMontgomery(baseField, gX);
  gY = field::maybeToMontgomery(baseField, gY);

  return ShortWeierstrassAttr::get(a.getContext(), baseField,
                                   cast<TypedAttr>(a), cast<TypedAttr>(b),
                                   cast<TypedAttr>(gX), cast<TypedAttr>(gY));
}

void ShortWeierstrassAttr::print(AsmPrinter &printer) const {
  Attribute a = field::maybeToStandard(getBaseField(), getA());
  Attribute b = field::maybeToStandard(getBaseField(), getB());
  Attribute gX = field::maybeToStandard(getBaseField(), getGx());
  Attribute gY = field::maybeToStandard(getBaseField(), getGy());

  printer << '<' << a << ',' << b << '(' << gX << ',' << gY
          << ")> : " << getBaseField();
}

} // namespace mlir::zkir::elliptic_curve
