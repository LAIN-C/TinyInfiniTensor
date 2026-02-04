#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    const size_t rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================

    IT_ASSERT(rank >= 1);
    IT_ASSERT(dims.size() == rank);

    // All inputs must have the same rank and identical dimensions except the
    // concatenation axis.
    int sumOnAxis = 0;
    for (auto &t : inputs) {
        IT_ASSERT(t->getRank() == rank);
        const auto cur = t->getDims();
        for (size_t i = 0; i < rank; ++i) {
            if ((int)i == dim)
                continue;
            IT_ASSERT(cur[i] == dims[i], "Concat: non-concat dims mismatch");
        }
        sumOnAxis += cur[dim];
    }
    dims[dim] = sumOnAxis;

    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
