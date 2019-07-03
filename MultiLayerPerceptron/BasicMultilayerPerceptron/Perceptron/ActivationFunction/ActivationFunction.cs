using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron.ActivationFunction
{
    public abstract class ActivationFunction
    {


        public static ActivationFunction ReLu()
        {
            return new ReLu();
        }

        public static ActivationFunction Sigmoid()
        {
            return new Sigmoid();
        }

        public static ActivationFunction SoftPlus()
        {
            return new SoftPlus();
        }

        public static ActivationFunction Tanh()
        {
            return new Tanh();
        }

        public abstract double GetValue(double x);

        public abstract double GetDerivativeValue(double x);




    }
}
