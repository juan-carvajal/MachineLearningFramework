using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron.ErrorFunction
{
    public abstract class ErrorFunction
    {
        public static ErrorFunction MSE()
        {
            return new MeanSquaredError();
        }


        public static ErrorFunction CrossEntropy()
        {
            return new CrossEntropy();
        }

        public abstract double GetValue(double label, double estimator);
        public abstract double GetDerivativeValue(double label, double estimator);
    }
}
