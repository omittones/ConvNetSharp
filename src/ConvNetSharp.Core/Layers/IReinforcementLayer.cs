using System;

namespace ConvNetSharp.Core.Layers
{
    public interface IReinforcementLayer<T> : ILastLayer<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        void SetLoss(int[][] pathActions, T[] loss);
    }
}
