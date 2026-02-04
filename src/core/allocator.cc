#include "core/allocator.h"
#include <algorithm>
#include <utility>

namespace infini
{
    namespace
    {
        // Insert/erase helpers keep the two indices consistent.
        inline void insertFreeBlock(std::map<size_t, size_t> &byAddr,
                                    std::map<size_t, size_t> &byEnd,
                                    size_t start, size_t size)
        {
            IT_ASSERT(size > 0);
            auto [it, inserted] = byAddr.emplace(start, size);
            IT_ASSERT(inserted, "Duplicated free block start");
            auto [it2, inserted2] = byEnd.emplace(start + size, start);
            IT_ASSERT(inserted2, "Duplicated free block end");
        }

        inline void eraseFreeBlock(std::map<size_t, size_t> &byAddr,
                                   std::map<size_t, size_t> &byEnd,
                                   size_t start)
        {
            auto it = byAddr.find(start);
            IT_ASSERT(it != byAddr.end(), "Free block not found");
            auto size = it->second;
            byAddr.erase(it);
            auto it2 = byEnd.find(start + size);
            IT_ASSERT(it2 != byEnd.end(), "Free block end not found");
            byEnd.erase(it2);
        }
    } // namespace

    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;
        heapEnd = 0;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        //
        // Strategy (simple & deterministic):
        // - First-fit on the free list ordered by address.
        // - Split a free block if it is larger than requested.
        // - If no block fits, extend the heap (bump `heapEnd`).
        //
        // We simulate allocation here (no real malloc). Actual memory is
        // allocated once in getPtr() with size == `peak`.
        // =================================== 作业 ===================================

        // 1) Try reuse a previously freed block.
        for (auto it = freeBlocksByAddr.begin(); it != freeBlocksByAddr.end();
             ++it)
        {
            const size_t blockStart = it->first;
            const size_t blockSize = it->second;
            if (blockSize < size)
                continue;

            // Consume from the head of this free block.
            eraseFreeBlock(freeBlocksByAddr, freeBlocksByEnd, blockStart);

            const size_t remain = blockSize - size;
            if (remain > 0)
            {
                insertFreeBlock(freeBlocksByAddr, freeBlocksByEnd,
                                blockStart + size, remain);
            }

            used += size;
            return blockStart;
        }

        // 2) Allocate at the end of the simulated heap.
        const size_t addr = heapEnd;
        heapEnd += size;
        used += size;
        peak = std::max(peak, heapEnd);
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        //
        // Strategy:
        // - Insert the freed interval into the free list.
        // - Coalesce with adjacent free blocks (both sides).
        // - If the resulting free block touches the heap end, shrink `heapEnd`
        //   and repeatedly pop any free blocks now at the end.
        //
        // This behavior is required by the unit test where the last allocated
        // block is freed and then a larger block is allocated at the same
        // offset.
        // =================================== 作业 ===================================

        IT_ASSERT(used >= size, "Allocator::free underflow");
        used -= size;

        size_t start = addr;
        size_t blockSize = size;

        // Merge with previous block if its end equals start.
        if (auto itPrev = freeBlocksByEnd.find(start);
            itPrev != freeBlocksByEnd.end())
        {
            const size_t prevStart = itPrev->second;
            // prev size is stored in freeBlocksByAddr
            auto itPrevSize = freeBlocksByAddr.find(prevStart);
            IT_ASSERT(itPrevSize != freeBlocksByAddr.end());
            const size_t prevSize = itPrevSize->second;
            eraseFreeBlock(freeBlocksByAddr, freeBlocksByEnd, prevStart);
            start = prevStart;
            blockSize += prevSize;
        }

        // Merge with next block if its start equals end.
        if (auto itNext = freeBlocksByAddr.find(start + blockSize);
            itNext != freeBlocksByAddr.end())
        {
            const size_t nextStart = itNext->first;
            const size_t nextSize = itNext->second;
            eraseFreeBlock(freeBlocksByAddr, freeBlocksByEnd, nextStart);
            blockSize += nextSize;
        }

        // Insert merged free block.
        insertFreeBlock(freeBlocksByAddr, freeBlocksByEnd, start, blockSize);

        // Try shrink the heap end if free blocks appear at the end.
        auto shrinkTail = [&]()
        {
            while (true)
            {
                auto itTail = freeBlocksByEnd.find(heapEnd);
                if (itTail == freeBlocksByEnd.end())
                    break;
                const size_t tailStart = itTail->second;
                eraseFreeBlock(freeBlocksByAddr, freeBlocksByEnd, tailStart);
                heapEnd = tailStart;
            }
        };

        // If the newly freed (possibly merged) block reaches the heap end,
        // shrinkTail() will pop it (and other tail blocks) and move heapEnd.
        shrinkTail();
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
