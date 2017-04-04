using System;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    /// <summary>
    ///     This is a classifier, with N discrete classes from 0 to N-1
    ///     it gets a stream of N incoming numbers and computes the softmax
    ///     function (exponentiate and normalize to sum to 1 as probabilities should)
    /// </summary>
    [DataContract]
    [Serializable]
    public class SoftmaxLayer : LastLayerBase, IClassificationLayer
    {
        [DataMember]
        private double[] es;

        public SoftmaxLayer(int classCount)
        {
            this.ClassCount = classCount;
        }

        [DataMember]
        public int ClassCount { get; set; }

        public override double Backward(double y)
        {
            var classIndex = (int)y;

            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.InputActivation;
            x.ZeroGradients(); // zero out the gradient of input Vol

            for (var i = 0; i < this.OutputDepth; i++)
            {
                var indicator = i == classIndex ? 1.0 : 0.0;
                var mul = -(indicator - this.es[i]);
                x.SetGradient(i, mul);
            }

            // loss is the class negative log likelihood
            return -Math.Log(this.es[classIndex]);
        }

        public override double Backward(double[] y)
        {
            var x = this.InputActivation;
            x.ZeroGradients();

            var loss = 0.0;
            for (var i = 0; i < this.OutputDepth; i++)
            {
                var grad = -(y[i] - this.es[i]);
                x.SetGradient(i, grad);

                double v;
                if (this.es[i] < double.Epsilon)
                    v = Math.Log(double.Epsilon);
                else
                    v = Math.Log(this.es[i]);

                loss += -(this.es[i]*v);
            }

            return loss;
        }

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;

            var outputActivation = new Volume(1, 1, this.OutputDepth, 0.0);

            // compute max activation
            var maxInput = input.Get(0);
            for (var i = 1; i < this.OutputDepth; i++)
                if (input.Get(i) > maxInput)
                    maxInput = input.Get(i);

            // compute exponentials (carefully to not blow up)
            var outputAct = new double[this.OutputDepth];
            var outputSum = 0.0;
            for (var i = 0; i < this.OutputDepth; i++)
            {
                var exp = Math.Exp(input.Get(i) - maxInput);
                outputSum += exp;
                outputAct[i] = exp;
            }

            // normalize and output to sum to one
            for (var i = 0; i < this.OutputDepth; i++)
            {
                outputAct[i] /= outputSum;
                outputActivation.Set(i, outputAct[i]);
            }

            this.es = outputAct; // save these for backprop
            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }
        
        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            var inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputDepth = inputCount;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
        }
    }
}