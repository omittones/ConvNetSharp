using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    public static class Assert<T>
        where T : struct, IEquatable<T>
    {
        public static void AreEqual(float expected, T actual)
        {
            Assert.AreEqual(Ops<T>.Cast(expected), actual);
        }
    }
}