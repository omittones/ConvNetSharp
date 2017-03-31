﻿using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class SoftmaxLayer<T> : LastLayerBase<T>, IClassificationLayer where T : struct, IEquatable<T>, IFormattable
    {
        public SoftmaxLayer(Dictionary<string, object> data) : base(data)
        {
            this.ClassCount = Convert.ToInt32(data["ClassCount"]);
        }

        public SoftmaxLayer(int classCount)
        {
            this.ClassCount = classCount;
        }

        public int ClassCount { get; set; }

        public override void Backward(Volume<T> y, out T loss)
        {
            this.OutputActivation.DoSoftMaxGradient(this.OutputActivation - y, this.InputActivationGradients);

            //loss is the class negative log likelihood
            loss = Ops<T>.Zero;
            for (var n = 0; n < y.Shape.GetDimension(3); n++)
            {
                for (var d = 0; d < y.Shape.GetDimension(2); d++)
                {
                    for (var h = 0; h < y.Shape.GetDimension(1); h++)
                    {
                        for (var w = 0; w < y.Shape.GetDimension(0); w++)
                        {
                            var current = Ops<T>.Multiply(y.Get(w, h, d, n),
                                Ops<T>.Log(this.OutputActivation.Get(w, h, d, n)));
                            loss = Ops<T>.Add(loss, current);
                        }
                    }
                }
            }

            loss = Ops<T>.Negate(loss);
        }

        public override void Backward(Volume<T> outputGradient)
        {
            throw new NotImplementedException();
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.DoSoftMax(this.OutputActivation);
            return this.OutputActivation;
        }

        public override Dictionary<string, object> GetData()
        {
            var dico = base.GetData();
            dico["ClassCount"] = this.ClassCount;
            return dico;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            var inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
            this.OutputDepth = inputCount;
        }
    }
}