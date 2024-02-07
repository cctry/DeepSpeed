// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once
#include <cutlass/epilogue/threadblock/default_epilogue_simt.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h>
#include "../gemm_kernel_utils.h"
#include "../iterators/predicated_tile_iterator_atomic.h"
#include "cutlass/epilogue/threadblock/epilogue.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {
template <int Rank,
          typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess>
struct EpilogueTensorOpAffineRankN : public DefaultEpilogueTensorOpAffineRankN<Rank,
                                                                               Shape_,
                                                                               WarpMmaTensorOp_,
                                                                               PartitionsK,
                                                                               OutputOp_,
                                                                               ElementsPerAccess> {
    using Base = DefaultEpilogueTensorOpAffineRankN<Rank,
                                                    Shape_,
                                                    WarpMmaTensorOp_,
                                                    PartitionsK,
                                                    OutputOp_,
                                                    ElementsPerAccess>;
    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileIteratorAffineRankNAtomic<
            typename Base::OutputTileThreadMap,
            typename Base::ElementOutput,
            Rank>;

    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaTensorOp,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding,
                                                 Base::kFragmentsPerIteration>;
};

template <int Rank,
          typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess>
struct EpilogueVoltaTensorOpAffineRankN
    : public DefaultEpilogueVoltaTensorOpAffineRankN<Rank,
                                                     Shape_,
                                                     WarpMmaTensorOp_,
                                                     PartitionsK,
                                                     OutputOp_,
                                                     ElementsPerAccess> {
    using Base = DefaultEpilogueVoltaTensorOpAffineRankN<Rank,
                                                         Shape_,
                                                         WarpMmaTensorOp_,
                                                         PartitionsK,
                                                         OutputOp_,
                                                         ElementsPerAccess>;
    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileIteratorAffineRankNAtomic<
            typename Base::OutputTileThreadMap,
            typename Base::ElementOutput,
            Rank>;

    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaTensorOp,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding>;
};

template <int Rank,
          typename Shape_,
          typename WarpMmaSimt_,
          typename OutputOp_,
          int ElementsPerAccess>
struct EpilogueSimtAffineRankN : public DefaultEpilogueSimtAffineRankN<Rank,
                                                                       Shape_,
                                                                       WarpMmaSimt_,
                                                                       OutputOp_,
                                                                       ElementsPerAccess> {
    using Base =
        DefaultEpilogueSimtAffineRankN<Rank, Shape_, WarpMmaSimt_, OutputOp_, ElementsPerAccess>;

    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileIteratorAffineRankNAtomic<
            typename Base::OutputTileThreadMap,
            typename Base::ElementOutput,
            Rank>;

    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaSimt,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding>;
};

template <typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct EpilogueTensorOp : public DefaultEpilogueTensorOp<Shape_,
                                                         WarpMmaTensorOp_,
                                                         PartitionsK,
                                                         OutputOp_,
                                                         ElementsPerAccess,
                                                         ScatterD,
                                                         PermuteDLayout> {
    using Base = DefaultEpilogueTensorOp<Shape_,
                                         WarpMmaTensorOp_,
                                         PartitionsK,
                                         OutputOp_,
                                         ElementsPerAccess,
                                         ScatterD,
                                         PermuteDLayout>;
    using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorAtomic<
        typename Base::OutputTileThreadMap,
        typename Base::ElementOutput,
        ScatterD,
        PermuteDLayout>;
    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaTensorOp,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding,
                                                 Base::kFragmentsPerIteration>;
};

template <typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct EpilogueVoltaTensorOp : public DefaultEpilogueVoltaTensorOp<Shape_,
                                                                   WarpMmaTensorOp_,
                                                                   PartitionsK,
                                                                   OutputOp_,
                                                                   ElementsPerAccess,
                                                                   ScatterD,
                                                                   PermuteDLayout> {
    using Base = DefaultEpilogueVoltaTensorOp<Shape_,
                                              WarpMmaTensorOp_,
                                              PartitionsK,
                                              OutputOp_,
                                              ElementsPerAccess,
                                              ScatterD,
                                              PermuteDLayout>;
    using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorAtomic<
        typename Base::OutputTileThreadMap,
        typename Base::ElementOutput,
        ScatterD,
        PermuteDLayout>;
    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaTensorOp,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding>;
};

template <typename Shape_,
          typename WarpMmaSimt_,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct EpilogueSimt : public DefaultEpilogueSimt<Shape_,
                                                 WarpMmaSimt_,
                                                 OutputOp_,
                                                 ElementsPerAccess,
                                                 ScatterD,
                                                 PermuteDLayout> {
    using Base = DefaultEpilogueSimt<Shape_,
                                     WarpMmaSimt_,
                                     OutputOp_,
                                     ElementsPerAccess,
                                     ScatterD,
                                     PermuteDLayout>;
    using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorAtomic<
        typename Base::OutputTileThreadMap,
        typename Base::ElementOutput,
        ScatterD,
        PermuteDLayout>;

    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaSimt,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding>;
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

template <typename Arch_,
          typename Shape_,
          typename DefaultGemm_,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = cutlass::layout::NoPermute,
          typename Enable = void>
struct BiasGradEpilogue {
    using WarpMmaTensorOp_ = typename DefaultGemm_::Mma::Operator;
    static constexpr int PartitionsK = DefaultGemm_::kPartitionsK;
    using Epilogue =
        typename cutlass::epilogue::threadblock::EpilogueTensorOp<Shape_,
                                                                  WarpMmaTensorOp_,
                                                                  PartitionsK,
                                                                  OutputOp_,
                                                                  ElementsPerAccess,
                                                                  ScatterD,
                                                                  PermuteDLayout>::Epilogue;
};

template <typename Mma, typename Arch>
constexpr bool is_simt_v = is_simt<Mma, Arch>::value;

template <typename Shape_,
          typename DefaultGemm_,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD,
          typename PermuteDLayout>
struct BiasGradEpilogue<
    cutlass::arch::Sm70,
    Shape_,
    DefaultGemm_,
    OutputOp_,
    ElementsPerAccess,
    ScatterD,
    PermuteDLayout,
    typename cutlass::platform::enable_if<!is_simt_v<DefaultGemm_, cutlass::arch::Sm70>>::type> {
    using WarpMmaTensorOp_ = typename DefaultGemm_::Mma::Operator;
    static constexpr int PartitionsK = DefaultGemm_::kPartitionsK;
    using Epilogue =
        typename cutlass::epilogue::threadblock::EpilogueVoltaTensorOp<Shape_,
                                                                       WarpMmaTensorOp_,
                                                                       PartitionsK,
                                                                       OutputOp_,
                                                                       ElementsPerAccess,
                                                                       ScatterD,
                                                                       PermuteDLayout>::Epilogue;
};

// SIMT version
template <typename Arch_,
          typename Shape_,
          typename DefaultGemm_,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD,
          typename PermuteDLayout>
struct BiasGradEpilogue<
    Arch_,
    Shape_,
    DefaultGemm_,
    OutputOp_,
    ElementsPerAccess,
    ScatterD,
    PermuteDLayout,
    typename cutlass::platform::enable_if<is_simt_v<DefaultGemm_, Arch_>>::type> {
    using WarpMmaTensorOp_ = typename DefaultGemm_::Mma::Operator;
    using Epilogue =
        typename cutlass::epilogue::threadblock::EpilogueSimt<Shape_,
                                                              WarpMmaTensorOp_,
                                                              OutputOp_,
                                                              ElementsPerAccess,
                                                              ScatterD,
                                                              PermuteDLayout>::Epilogue;
};

template <typename Arch_,
          int Rank,
          typename Shape_,
          typename DefaultGemm_,
          typename OutputOp_,
          int ElementsPerAccess,
          typename Enable = void>
struct BiasGradEpilogueAffineRankN {
    using WarpMmaTensorOp_ = typename DefaultGemm_::Mma::Operator;
    static constexpr int PartitionsK = DefaultGemm_::kPartitionsK;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueTensorOpAffineRankN<
        Rank,
        Shape_,
        WarpMmaTensorOp_,
        PartitionsK,
        OutputOp_,
        ElementsPerAccess>::Epilogue;
};

template <int Rank,
          typename Shape_,
          typename DefaultGemm_,
          typename OutputOp_,
          int ElementsPerAccess>
struct BiasGradEpilogueAffineRankN<
    cutlass::arch::Sm70,
    Rank,
    Shape_,
    DefaultGemm_,
    OutputOp_,
    ElementsPerAccess,
    typename cutlass::platform::enable_if<!is_simt_v<DefaultGemm_, cutlass::arch::Sm70>>::type> {
    using WarpMmaTensorOp_ = typename DefaultGemm_::Mma::Operator;
    static constexpr int PartitionsK = DefaultGemm_::kPartitionsK;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueVoltaTensorOpAffineRankN<
        Rank,
        Shape_,
        WarpMmaTensorOp_,
        PartitionsK,
        OutputOp_,
        ElementsPerAccess>::Epilogue;
};

// SIMT version
template <typename Arch_,
          int Rank,
          typename Shape_,
          typename DefaultGemm_,
          typename OutputOp_,
          int ElementsPerAccess>
struct BiasGradEpilogueAffineRankN<
    Arch_,
    Rank,
    Shape_,
    DefaultGemm_,
    OutputOp_,
    ElementsPerAccess,
    typename cutlass::platform::enable_if<is_simt_v<DefaultGemm_, Arch_>>::type> {
    using WarpMmaTensorOp_ = typename DefaultGemm_::Mma::Operator;
    using Epilogue = typename cutlass::epilogue::threadblock::
        EpilogueSimtAffineRankN<Rank, Shape_, WarpMmaTensorOp_, OutputOp_, ElementsPerAccess>::
            Epilogue;
};
