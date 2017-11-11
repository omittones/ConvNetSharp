using System;
using ConvNetSharp.Core.Layers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    public abstract class IntegrationTests<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        public abstract double Epsilon { get; }

        public virtual void Minimal()
        {
            var net = new Net<T>();
            net.AddLayer(new InputLayer<T>(3, 3, 3));
            net.AddLayer(new FullyConnLayer<T>(3));
            net.AddLayer(new SoftmaxLayer<T>());

            GradientCheckTools.CheckGradientOnNet(net, epsilon: Epsilon);
        }

        public virtual void FeedForwardRelu1Hidden()
        {
            var net = new Net<T>();
            net.AddLayer(new InputLayer<T>(3, 3, 3));
            net.AddLayer(new FullyConnLayer<T>(10));
            net.AddLayer(new ReluLayer<T>());
            net.AddLayer(new FullyConnLayer<T>(5));
            net.AddLayer(new SoftmaxLayer<T>());

            GradientCheckTools.CheckGradientOnNet(net, epsilon: Epsilon);
        }
    }
}