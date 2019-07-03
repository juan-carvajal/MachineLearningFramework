using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron.ErrorFunction
{
    class MeanSquaredError : ErrorFunction
    {
        public override double GetDerivativeValue(double label, double estimator)
        {
            return 2.0 * (estimator - label);
        }

        public override double GetValue(double label, double estimator)
        {
            return Math.Pow(estimator - label, 2);
        }
    }
}
