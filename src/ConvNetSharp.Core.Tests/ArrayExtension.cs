using System;
using System.Linq;

namespace ConvNetSharp.Core.Tests
{
    public static class ArrayExtension
    {
        public static T[] To<T>(this double[] input)
            where T : struct, IEquatable<T>
        {
            return input
                .Select(i => Convert.ChangeType(i, typeof(T)))
                .Cast<T>()
                .ToArray();
        }
    }
}