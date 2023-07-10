#pragma once
#include "../iterators/predicated_tile_iterator_atomic.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>

namespace cutlass {
namespace epilogue {
namespace threadblock {
template <int Rank, typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
          typename OutputOp_, int ElementsPerAccess>
struct EpilogueTensorOpAffineRankN
    : public DefaultEpilogueTensorOpAffineRankN<Rank, Shape_, WarpMmaTensorOp_,
                                                PartitionsK, OutputOp_,
                                                ElementsPerAccess> {
  using Base = DefaultEpilogueTensorOpAffineRankN<Rank, Shape_,
                                                  WarpMmaTensorOp_, PartitionsK,
                                                  OutputOp_, ElementsPerAccess>;
  using OutputTileIterator =
      cutlass::epilogue::threadblock::PredicatedTileIteratorAffineRankNAtomic<
          typename Base::OutputTileThreadMap, typename Base::ElementOutput,
          Rank>;

  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      typename Base::Shape, typename Base::WarpMmaTensorOp, Base::kPartitionsK,
      OutputTileIterator, typename Base::AccumulatorFragmentIterator,
      typename Base::WarpTileIterator, typename Base::SharedLoadIterator,
      typename Base::OutputOp, typename Base::Padding,
      Base::kFragmentsPerIteration>;
};

template <typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
          typename OutputOp_, int ElementsPerAccess, bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct EpilogueTensorOp
    : public DefaultEpilogueTensorOp<Shape_, WarpMmaTensorOp_, PartitionsK,
                                     OutputOp_, ElementsPerAccess, ScatterD,
                                     PermuteDLayout> {
  using Base =
      DefaultEpilogueTensorOp<Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_,
                              ElementsPerAccess, ScatterD, PermuteDLayout>;
  using OutputTileIterator =
      cutlass::epilogue::threadblock::PredicatedTileIteratorAtomic<
          typename Base::OutputTileThreadMap, typename Base::ElementOutput,
          ScatterD, PermuteDLayout>;
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      typename Base::Shape, typename Base::WarpMmaTensorOp, Base::kPartitionsK,
      OutputTileIterator, typename Base::AccumulatorFragmentIterator,
      typename Base::WarpTileIterator, typename Base::SharedLoadIterator,
      typename Base::OutputOp, typename Base::Padding,
      Base::kFragmentsPerIteration>;
};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass