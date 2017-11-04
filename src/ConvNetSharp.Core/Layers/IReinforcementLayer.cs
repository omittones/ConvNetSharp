using System;

namespace ConvNetSharp.Core.Layers
{
    public interface IReinforcementLayer<T> : ILastLayer<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        void SetReturns(int[][] pathActions, T[] returns, T baseline);
    }
}
