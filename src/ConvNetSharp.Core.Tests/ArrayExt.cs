using System;
using System.Linq;

namespace ConvNetSharp.Core.Tests
{
    public static class ArrayExt
    {
        public static T[] To<T>(this float[] input)
            where T : struct, IEquatable<T>
        {
            return input
                .Select(Ops<T>.Cast)
                .ToArray();
        }
    }
}