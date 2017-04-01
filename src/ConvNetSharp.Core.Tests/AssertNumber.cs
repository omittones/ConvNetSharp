using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    public static class AssertNumber
    {
        public static void AreEqual<T>(double expected, T actual, double delta = 0)
        {
            var value = (double) Convert.ChangeType(actual, typeof(double));
            Assert.AreEqual(expected, value, delta);
        }
    }
}