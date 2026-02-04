#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            // Default behavior in this project: identity permutation.
            // (Some frameworks default to reversing dims, but here we keep
            // existing semantics and make it safe by resizing the vector.)
            transposePermute.resize(rank);
            for (size_t i = 0; i < rank; ++i)
                transposePermute[i] = (int)i;
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        auto output_dim = input_dim;
        int rank = A->getRank();

        // =================================== 作业 ===================================
        // TODO：修改 output_dim，返回正确的 transpose 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21
        // =================================== 作业 ===================================

        IT_ASSERT((int)transposePermute.size() == rank);
        // ONNX-style permutation: output[i] = input[perm[i]].
        for (int i = 0; i < rank; ++i)
        {
            const int srcAxis = transposePermute[i];
            IT_ASSERT(srcAxis >= 0 && srcAxis < rank);
            output_dim[i] = input_dim[srcAxis];
        }

        return {{output_dim}};
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
