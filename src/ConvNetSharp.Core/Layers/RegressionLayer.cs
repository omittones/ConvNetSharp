using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    /// <summary>
    ///     implements an L2 regression cost layer,
    ///     so penalizes \sum_i(||x_i - y_i||^2), where x is its input
    ///     and y is the user-provided array of "correct" values.
    ///     Input should have a shape of [1, 1, 1, n] where n is the batch size
    /// </summary>
    public class RegressionLayer<T> : LastLayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> _result;
        private Volume<T> _sum;

        public RegressionLayer()
        {
        }

        public RegressionLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public override void Backward(Volume<T> y)
        {
            T unused;
            Backward(y, out unused);
        }

        public override void Backward(Volume<T> y, out T loss)
        {
            var yAdjusted = y.ReShape(new Shape(1, 1, -1, Shape.Keep));
            var inputActGrad = this.InputActivationGradients.ReShape(this.OutputActivation.Shape);
            yAdjusted.DoSubtractFrom(this.OutputActivation, inputActGrad);

            if (this._result == null ||
                this._result.Shape.GetDimension(3) != this.OutputActivation.Shape.GetDimension(3))
            {
                this._result = BuilderInstance<T>.Volume.SameAs(this.OutputActivation.Shape);
                this._sum = BuilderInstance<T>.Volume.SameAs(new Shape(1));
            }

            this._sum.Clear();
            inputActGrad.DoMultiply(inputActGrad, this._result); // dy * dy
            var half = Ops<T>.Cast(0.5);
            this._result.DoMultiply(this._result, half); // dy * dy * 0.5
            this._result.DoSum(this._sum); // sum over all batch

            var batchSize = y.Shape.GetDimension(3);
            loss = Ops<T>.Divide(this._sum.Get(0), Ops<T>.Cast(batchSize)); // average
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            return input;
        }
    }
}