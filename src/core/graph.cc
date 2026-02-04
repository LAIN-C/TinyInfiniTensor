#include "core/graph.h"
#include "core/blob.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        // The tests expect two kinds of optimizations:
        // 1) Transpose + Transpose cancellation when they are consecutive and
        //    their permutations are inverse of each other.
        // 2) Fuse a "swap-last-two-dims" Transpose into Matmul's transA/transB.
        //
        // Implementation note:
        // We only mutate operator input tensors and then *rebuild* all
        // graph connections (tensor source/targets, op pred/succ) to keep the
        // internal structure consistent.

        IT_ASSERT(topo_sort() == true);

        auto isInversePermute = [](const std::vector<int> &p,
                                   const std::vector<int> &q) -> bool
        {
            if (p.size() != q.size())
                return false;
            const int rank = static_cast<int>(p.size());
            std::vector<int> inv(rank, -1);
            for (int i = 0; i < rank; ++i)
            {
                IT_ASSERT(p[i] >= 0 && p[i] < rank);
                inv[p[i]] = i;
            }
            return inv == q;
        };

        auto isSwapLastTwoDims = [](const std::vector<int> &perm) -> bool
        {
            const int rank = static_cast<int>(perm.size());
            if (rank < 2)
                return false;
            for (int i = 0; i < rank - 2; ++i)
                if (perm[i] != i)
                    return false;
            return perm[rank - 2] == rank - 1 && perm[rank - 1] == rank - 2;
        };

        // Producer map: output tensor -> operator
        std::unordered_map<TensorObj *, Operator> producer;
        producer.reserve(tensors.size());
        for (auto &op : ops)
            for (auto &out : op->getOutputs())
                producer[out.get()] = op;

        std::unordered_set<OperatorObj *> removeOps;
        std::unordered_set<TensorObj *> removeTensors;

        // -------- Rule 1: cancel adjacent transpose pairs --------
        for (size_t i = 0; i + 1 < ops.size(); ++i)
        {
            auto t1 = as<TransposeObj>(ops[i]);
            auto t2 = as<TransposeObj>(ops[i + 1]);
            if (!t1 || !t2)
                continue;

            auto t1Out = t1->getOutput(0);
            auto t2In = t2->getInputs(0);
            if (t1Out != t2In)
                continue;

            const auto p1 = t1->getPermute();
            const auto p2 = t2->getPermute();
            if (!isInversePermute(p1, p2))
                continue;

            // Replace uses of t2's output by t1's input.
            auto src = t1->getInputs(0);
            auto dst = t2->getOutput(0);
            for (auto &op : ops)
            {
                if (removeOps.count(op.get()))
                    continue;
                op->replaceInput(dst, src);
            }

            removeOps.insert(t1.get());
            removeOps.insert(t2.get());
            removeTensors.insert(t1Out.get());
            removeTensors.insert(dst.get());
        }

        // -------- Rule 2: fuse transpose into matmul transA/transB --------
        for (auto &op : ops)
        {
            if (removeOps.count(op.get()))
                continue;

            auto mm = as<MatmulObj>(op);
            if (!mm)
                continue;

            for (int inputIdx = 0; inputIdx < 2; ++inputIdx)
            {
                auto inputTensor = mm->getInputs(inputIdx);
                auto itProd = producer.find(inputTensor.get());
                if (itProd == producer.end())
                    continue;
                if (removeOps.count(itProd->second.get()))
                    continue;

                auto tp = as<TransposeObj>(itProd->second);
                if (!tp)
                    continue;

                const auto perm = tp->getPermute();
                if (!isSwapLastTwoDims(perm))
                    continue;

                // Fuse: remove transpose by toggling the corresponding flag.
                // Example: Matmul(transpose(A), B) == Matmul(A, B, transA=true)
                auto original = tp->getInputs(0);
                mm->replaceInput(inputTensor, original);
                if (inputIdx == 0)
                    mm->setTransA(!mm->getTransA());
                else
                    mm->setTransB(!mm->getTransB());

                removeOps.insert(tp.get());
                removeTensors.insert(tp->getOutput(0).get());
            }
        }

        // -------- Materialize removals & prune unused tensors --------
        OpVec newOps;
        newOps.reserve(ops.size());
        for (auto &op : ops)
            if (!removeOps.count(op.get()))
                newOps.emplace_back(op);

        // Collect tensors that are still referenced by remaining ops.
        std::unordered_set<TensorObj *> usedTensorPtrs;
        usedTensorPtrs.reserve(tensors.size());
        for (auto &op : newOps)
        {
            for (auto &t : op->getInputs())
                usedTensorPtrs.insert(t.get());
            for (auto &t : op->getOutputs())
                usedTensorPtrs.insert(t.get());
        }

        TensorVec newTensors;
        newTensors.reserve(tensors.size());
        for (auto &t : tensors)
        {
            if (removeTensors.count(t.get()))
                continue;
            if (usedTensorPtrs.count(t.get()))
                newTensors.emplace_back(t);
        }

        // -------- Rebuild all graph connections --------
        tensors = std::move(newTensors);
        for (auto &t : tensors)
        {
            t->targets.clear();
            t->source.reset();
        }
        for (auto &op : newOps)
        {
            op->predecessors.clear();
            op->successors.clear();
        }

        ops.clear();
        sorted = false;
        for (auto &op : newOps)
            addOperatorAndConnect(op);
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        // We perform a simple lifetime-based memory planning:
        // - Graph input tensors must be allocated first so users can setData().
        // - As we traverse operators in topological order, we allocate outputs.
        // - We free an input tensor when it has been consumed by all its target
        //   operators (i.e., its last use), which enables memory reuse.
        //
        // Note: Tensors that are graph outputs (no targets) are never freed.

        std::unordered_map<TensorObj *, int> remainingUses;
        remainingUses.reserve(tensors.size());
        for (auto &t : tensors)
            remainingUses.emplace(t.get(), (int)t->getTargets().size());

        std::unordered_set<TensorObj *> keepAlive;
        for (auto &t : getOutputs())
            keepAlive.insert(t.get());

        std::unordered_map<TensorObj *, size_t> offsets;
        offsets.reserve(tensors.size());

        auto allocTensorIfNeeded = [&](const Tensor &t)
        {
            if (!t)
                return;
            if (offsets.find(t.get()) != offsets.end())
                return;
            offsets.emplace(t.get(), allocator.alloc(t->getBytes()));
        };

        // Allocate all graph inputs first (no source operator).
        for (auto &t : tensors)
            if (!t->getSource())
                allocTensorIfNeeded(t);

        // Allocate outputs and release dead inputs.
        for (auto &op : ops)
        {
            for (auto &out : op->getOutputs())
                allocTensorIfNeeded(out);

            for (auto &in : op->getInputs())
            {
                auto it = remainingUses.find(in.get());
                IT_ASSERT(it != remainingUses.end());
                it->second -= 1;
                if (it->second == 0 && keepAlive.count(in.get()) == 0)
                {
                    allocator.free(offsets.at(in.get()), in->getBytes());
                }
            }
        }

        // Perform one real allocation and bind each tensor to its planned offset.
        void *base = allocator.getPtr();
        for (auto &t : tensors)
        {
            auto it = offsets.find(t.get());
            IT_ASSERT(it != offsets.end(), "Tensor missing memory plan");
            auto ptr = static_cast<void *>(static_cast<char *>(base) + it->second);
            t->setDataBlob(make_ref<BlobObj>(runtime, ptr));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini