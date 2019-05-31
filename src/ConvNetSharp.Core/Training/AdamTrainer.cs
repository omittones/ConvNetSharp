using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Training
{
    public class AdamTrainer<T> : TrainerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly List<Volume<T>> maOfGrad = new List<Volume<T>>();
        private readonly List<Volume<T>> maOfSqGrad = new List<Volume<T>>();
        private readonly List<Volume<T>> temp1 = new List<Volume<T>>();
        private readonly List<Volume<T>> temp2 = new List<Volume<T>>();
        private readonly List<Volume<T>> gradGrad = new List<Volume<T>>();
        private readonly Volume<T> two;
        private Volume<T> epsilon;
        private T oldEpsilon;
        private int k;

        public AdamTrainer(INet<T> net) : base(net)
        {
            this.Eps = Ops<T>.Cast(1e-8);
            this.Beta1 = Ops<T>.Cast(0.9);
            this.Beta2 = Ops<T>.Cast(0.999);
            this.LearningRate = Ops<T>.One;

            this.oldEpsilon = this.Eps;
            this.two = BuilderInstance<T>.Volume.From(new[] { Ops<T>.Cast(2.0) }, new Shape(1));
            this.epsilon = BuilderInstance<T>.Volume.From(new[] { this.Eps }, new Shape(1));
        }

        public T Beta1 { get; set; }

        public T Beta2 { get; set; }

        public T LearningRate { get; set; }

        public T Eps { get; set; }

        protected override void TrainImplem(int batchSize)
        {
            var parametersAndGradients = this.Net
                .GetParametersAndGradients()
                .ToArray();

            // initialize lists for accumulators. Will only be done once on first iteration
            if (this.maOfGrad.Count == 0)
            {
                for (var i = 0; i < parametersAndGradients.Length; i++)
                {
                    var shape = parametersAndGradients[i].Volume.Shape;
                    this.maOfGrad.Add(BuilderInstance<T>.Volume.SameAs(shape));
                    this.maOfSqGrad.Add(BuilderInstance<T>.Volume.SameAs(shape));
                    this.temp1.Add(BuilderInstance<T>.Volume.SameAs(shape));
                    this.temp2.Add(BuilderInstance<T>.Volume.SameAs(shape));
                    this.gradGrad.Add(BuilderInstance<T>.Volume.SameAs(shape));
                }
            }
            else
            {
                for (var i = 0; i < parametersAndGradients.Length; i++)
                {
                    var shape = parametersAndGradients[i].Volume.Shape;
                    if (!this.maOfGrad[i].Shape.Equals(shape))
                    {
                        this.maOfGrad[i] = BuilderInstance<T>.Volume.SameAs(shape);
                        this.maOfSqGrad[i] = BuilderInstance<T>.Volume.SameAs(shape);
                        this.temp1[i] = BuilderInstance<T>.Volume.SameAs(shape);
                        this.temp2[i] = BuilderInstance<T>.Volume.SameAs(shape);
                        this.gradGrad[i] = BuilderInstance<T>.Volume.SameAs(shape);
                    }
                }
            }            

            if (!this.oldEpsilon.Equals(this.Eps))
            {
                this.epsilon.Set(0, this.Eps);
                this.oldEpsilon = this.Eps;
            }

            // perform an update for all sets of weights
            for (var i = 0; i < parametersAndGradients.Length; i++)
            {
                var parametersAndGradient = parametersAndGradients[i];
                var vol = parametersAndGradient.Volume;
                var grad = parametersAndGradient.Gradient;
                var temp1 = this.temp1[i];
                var temp2 = this.temp2[i];
                var gradGrad = this.gradGrad[i];

                grad.Multiply(Ops<T>.Divide(Ops<T>.One, Ops<T>.Cast(batchSize)), grad); // grad *= 1 / BatchSize

                // momentum update
                // update biased first moment estimate: gsum[i] = gsum[i] * Beta1 +  (1 - Beta1) * grad
                this.maOfGrad[i].Multiply(this.Beta1, temp1); // temp1 = this.gsum[i] * this.Beta1
                grad.Multiply(Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(this.Beta1)), this.maOfGrad[i]); //  this.gsum[i] =  grad * (1 - Beta1)
                temp1.Add(this.maOfGrad[i]); //  this.gsum[i] += temp1

                grad.Power(two, gradGrad); // gradgrad = grad * grad

                // update biased second moment estimate: xsum[i] = xsum[i] * Beta2 +  (1 - Beta2) * grad * grad
                this.maOfSqGrad[i].Multiply(this.Beta2, temp1); // temp1 = this.xsum[i] * this.Beta2
                gradGrad.Multiply(Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(this.Beta2)), this.maOfSqGrad[i]); // temp2 =  gradgrad * (1 - Beta2)
                temp1.Add(this.maOfSqGrad[i]); //  this.xsum[i] += temp1

                var biasCorr1 = temp1;
                var biasCorr2 = temp2;

                this.maOfGrad[i].Multiply(Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(Ops<T>.Pow(this.Beta1, Ops<T>.Cast(this.k)))), biasCorr1); // correct bias first moment estimate
                this.maOfSqGrad[i].Multiply(Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(Ops<T>.Pow(this.Beta2, Ops<T>.Cast(this.k)))), biasCorr2); // correct bias second moment estimate

                biasCorr2.Sqrt(biasCorr2); // biasCorr2 = sqrt(biasCorr2)
                epsilon.Add(biasCorr2); // biasCorr2 += epsilon

                var dx = biasCorr1;
                dx.Multiply(this.LearningRate, dx);
                dx.Divide(biasCorr2, dx);

                dx.SubtractFrom(vol, vol);

                grad.Clear(); // zero out gradient so that we can begin accumulating anew

                this.k += 1;
            }
        }
    }
}