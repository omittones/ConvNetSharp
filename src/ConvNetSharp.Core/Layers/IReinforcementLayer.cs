using System;
using ConvNetSharp.Core.Layers;

namespace ConvNetSharp.Core.Layers
{
    public interface IReinforcementLayer<T> : ILastLayer<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        void SetLoss(T[] loss);
    }
}
