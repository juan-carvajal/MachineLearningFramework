using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BasicMultilayerPerceptron.DataSet;
using BasicMultilayerPerceptron.Perceptron.ActivationFunction;

namespace BasicMultilayerPerceptron.Perceptron
{
    class Layer
    {

        internal double[,] WeightMatrix;
        internal double[] BiasVector;

        internal double[,] WeightMatrixChangeRecord;
        internal double[] BiasVectorChangeRecord;

        internal  ActivationFunction.ActivationFunction ActivationFunction;

        internal double[] Activations;
        internal double[] CostDerivatives;

        internal double[] WeightedSum;

        internal int Size;

        internal Layer Previous;
        internal Layer Next;


        public Layer(ActivationFunction.ActivationFunction activationFunction , int size)
        {
            this.ActivationFunction = activationFunction;
            this.Size = size;
            BiasVector = new double[size];
            Random rand = new Random();
            double prob = rand.NextDouble();
            for (int i = 0; i < BiasVector.Length; i++)
            {
                if (prob <= 0.5)
                {
                BiasVector[i] = -rand.NextDouble();

                }
                else
                {
                    BiasVector[i] = rand.NextDouble();
                }
            }
            BiasVectorChangeRecord = new double[size];
            Activations = new double[size];
            WeightedSum = new double[size];
            CostDerivatives = new double[size];
        }


        internal void BackPropagate(double error,double[]labels , ErrorFunction.ErrorFunction lossFunction)
        {
            for (int j = 0; j < CostDerivatives.Length; j++)
            {
                CostDerivatives[j] = lossFunction.GetDerivativeValue(labels[j], Activations[j]);
                //2.0 * (this.Layers[l].Activations[j] - labels[j]);
            }

            for (int j = 0; j < Activations.Length; j++)
            {
                BiasVectorChangeRecord[j] += ActivationFunction.GetDerivativeValue(WeightedSum[j]) * CostDerivatives[j];
                for (int k = 0; k < WeightMatrix.GetLength(1); k++)
                {
                    WeightMatrixChangeRecord[j, k] += Previous.Activations[k]
                        * ActivationFunction.GetDerivativeValue(WeightedSum[j])
                        * CostDerivatives[j];
                }
            }

            
        }


        private void BackPropagate()
        {
            for (int j = 0; j < CostDerivatives.Length; j++)
            {
                //this.Layers[l].CostDerivatives[j] = 2.0 * (this.Layers[l].Activations[j] - labels[j]);
                double acum = 0;
                for (int j2 = 0; j2 < Next.Size; j2++)
                {
                    acum += Next.WeightMatrix[j2, j] * Next.ActivationFunction.GetDerivativeValue(Next.WeightedSum[j2]) * Next.CostDerivatives[j2];
                }
                CostDerivatives[j] = acum;
            }

            for (int j = 0; j < Activations.Length; j++)
            {
                BiasVectorChangeRecord[j] += ActivationFunction.GetDerivativeValue(WeightedSum[j]) * CostDerivatives[j];
                for (int k = 0; k < WeightMatrix.GetLength(1); k++)
                {
                    WeightMatrixChangeRecord[j, k] += Previous.Activations[k]
                        * ActivationFunction.GetDerivativeValue(WeightedSum[j])
                        * CostDerivatives[j];
                }
            }


            if (Previous != null && Previous.Previous != null)
            {
                Previous.BackPropagate();

            }
        }





        public void BuildWeightMatrix()
        {
            Random rand = new Random();
            double prob = rand.NextDouble();
            WeightMatrix = new double[Size, Previous.Size];
            for (int i = 0; i < WeightMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < WeightMatrix.GetLength(1); j++)
                {
                    if (prob <= 0.5)
                    {
                    WeightMatrix[i, j] = -rand.NextDouble();

                    }
                    else
                    {
                        WeightMatrix[i, j] = rand.NextDouble();
                    }
                }
            }
            WeightMatrixChangeRecord = new double[Size, Previous.Size];
        }


        public void Activate(double[] data)
        {
            if(data.Length != Activations.Length)
            {
                throw new ArgumentException("Provided data should be of same size as Activations.");
            }
            else
            {
                
                Activations = data;
                Next.Activate();
            }
        }

        public void Activate()
        {
            var W = this.WeightMatrix;
            var A = Previous.Activations;

            double[] result = new double[Size];
            for (int i = 0; i < W.GetLength(0); i++)
            {
                for (int j = 0; j < W.GetLength(1); j++)
                {
                    result[i] += W[i,j] * A[j];
                }
                result[i] += BiasVector[i];
                this.WeightedSum[i] = result[i];
                result[i] = ActivationFunction.GetValue(result[i]);
            }
            Activations = result;
            if (Next != null)
            {

            Next.Activate();
            }
        }


        public double[] GetActivations()
        {
            return Activations.Clone() as double[];
        }

    }
}
