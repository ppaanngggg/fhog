#include "fhog.h"

using namespace std;
using namespace fhog;

double eps = 0.0001;

// Unit vectors used to compute gradient orientation
float uu[9] =
{
    1.0000,
    0.9397,
    0.7660,
    0.500,
    0.1736,
    -0.1736,
    -0.5000,
    -0.7660,
    -0.9397
};

float vv[9] =
{
    0.0000,
    0.3420,
    0.6428,
    0.8660,
    0.9848,
    0.9848,
    0.8660,
    0.6428,
    0.3420
};

cv::Mat fhog::fhog( const cv::Mat_<float> &mximage, const int sbin)
{
    mximage /= 255.;
	int dims[3] = {mximage.rows, mximage.cols, mximage.channels()};
	float im[dims[0] * dims[1] * 3];
	int imgSize = dims[0] * dims[1];
	int count = 0;
	for (int col = 0; col < dims[1]; col++) {
		for (int row = 0; row < dims[0]; row++) {
            if (dims[2] == 1) {
                float pixel = mximage.at<float>(row, col);
                im[count] = im[count + imgSize] = im[count + 2 * imgSize] = pixel;
            }
            if (dims[2] == 3) {
                cv::Vec3f pixel = mximage.at<cv::Vec3f>(row, col);
                im[count] = pixel[0];
                im[count + imgSize] = pixel[1];
                im[count + 2 * imgSize] = pixel[2];
            }
            count++;
		}
	}

	// memory for caching orientation histograms & their norms
	int blocks[2];
	blocks[0] = (int)round((float)dims[0]/(float)sbin);
	blocks[1] = (int)round((float)dims[1]/(float)sbin);

	assert (blocks[0] > 0);
	assert (blocks[1] > 0);

	float *hist = new float [blocks[0]*blocks[1]*18];

	for (int i = 0; i < blocks[0]*blocks[1]*18; i++)
		hist[i] = 0;

	float *norm = new float [blocks[0]*blocks[1]];

	for (int i = 0; i < blocks[0]*blocks[1]; i++)
		norm[i] = 0;

	// memory for HOG features
	int out[3];

	out[0] = max(blocks[0]-2, 0);
	out[1] = max(blocks[1]-2, 0);
	out[2] = 27+4+1;

	//CvMatND *mxfeat;
	cv::Mat mxfeat;
	//cout << "POS" << endl;
	if (out[0] == 0 && out[1] == 0 )
	{
		cout << "Empty matrix" << endl;
		mxfeat = 0.0;
	}
	else
	{
		assert (out[0] > 0);
		assert (out[1] > 0);

		mxfeat.create(out[0], out[1], CV_64FC(out[2]));
		assert (!mxfeat.empty());
		//cout << "POS" << endl;
		float *feat = new float [out[0] * out[1] * out[2]];
		assert (feat != NULL);
		//cout << "POS" << endl;
		int visible[2];
		visible[0] = blocks[0]*sbin;
		visible[1] = blocks[1]*sbin;

		float *s;
		float dy, dx;
		float v;

		float dy2, dx2;
		float v2;

		float dy3, dx3;
		float v3;

		float best_dot = 0;
		int best_o = 0;

		float dot;

		float xp, yp;

		int ixp, iyp;
		float vx0, vy0, vx1, vy1;

		float *src1, *src2;
		float *dst, *end;

		float *dst2;
		float *src, *p, n1, n2, n3, n4;

		float t1 = 0;
		float t2 = 0;
		float t3 = 0;
		float t4 = 0;

		double h1, h2, h3, h4;
		//
		//
		//
		float sum = 0;//
		//
		for (int x = 1; x < visible[1]-1; x++)
		{
			for (int y = 1; y < visible[0]-1; y++)
			{
				// first color channel
				s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
				dy = *(s+1) - *(s-1);
				dx = *(s+dims[0]) - *(s-dims[0]);
				v = dx*dx + dy*dy;

				// second color channel
				s += dims[0]*dims[1];
				dy2 = *(s+1) - *(s-1);
				dx2 = *(s+dims[0]) - *(s-dims[0]);
				v2 = dx2*dx2 + dy2*dy2;

				// third color channel
				s += dims[0]*dims[1];
				dy3 = *(s+1) - *(s-1);
				dx3 = *(s+dims[0]) - *(s-dims[0]);
				v3 = dx3*dx3 + dy3*dy3;

				// pick channel with strongest gradient
				if (v2 > v)
				{
					v = v2;
					dx = dx2;
					dy = dy2;
				}

				if (v3 > v)
				{
					v = v3;
					dx = dx3;
					dy = dy3;
				}

				// snap to one of 18 orientations
				best_dot = 0;
				best_o = 0;

				for (int o = 0; o < 9; o++)
				{
					dot = uu[o]*dx + vv[o]*dy;
					if (dot > best_dot)
					{
						best_dot = dot;
						best_o = o;
					}

					else if (-dot > best_dot)
					{
						best_dot = -dot;
						best_o = o+9;
					}
				}

				// add to 4 histograms around pixel using linear interpolation
				xp = ((float)x+0.5)/(float)sbin - 0.5;
				yp = ((float)y+0.5)/(float)sbin - 0.5;

				ixp = (int)floor(xp);
				iyp = (int)floor(yp);
				vx0 = xp-ixp;
				vy0 = yp-iyp;
				vx1 = 1.0-vx0;
				vy1 = 1.0-vy0;
				v = sqrt(v);
				//cout << "POS" << endl;
				if (ixp >= 0 && iyp >= 0)
				{
					*(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += vx1*vy1*v;
				}

				if (ixp+1 < blocks[1] && iyp >= 0)
				{
					*(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += vx0*vy1*v;
				}

				if (ixp >= 0 && iyp+1 < blocks[0])
				{
					*(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += vx1*vy0*v;
				}

				if (ixp+1 < blocks[1] && iyp+1 < blocks[0])
				{
					*(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += vx0*vy0*v;
				}
			}
		}
		//cout << "POS" << endl;
		// compute energy in each block by summing over orientations
		for (int o = 0; o < 9; o++)
		{
			src1 = hist + o*blocks[0]*blocks[1];
			src2 = hist + (o+9)*blocks[0]*blocks[1];
			dst = norm;
			end = norm + blocks[1]*blocks[0];

			while (dst < end)
			{
				*(dst++) += (*src1 + *src2) * (*src1 + *src2);
				src1++;
				src2++;
			}
		}
		//cout << "POS" << endl;
		// compute features
		for (int x = 0; x < out[1]; x++)
		{
			for (int y = 0; y < out[0]; y++)
			{
				dst2 = feat + x*out[0] + y;

				p = norm + (x+1)*blocks[0] + y+1;
				n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
				p = norm + (x+1)*blocks[0] + y;
				n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
				p = norm + x*blocks[0] + y+1;
				n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
				p = norm + x*blocks[0] + y;
				n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

				t1 = 0;
				t2 = 0;
				t3 = 0;
				t4 = 0;

				// contrast-sensitive features
				src = hist + (x+1)*blocks[0] + (y+1);

				for (int o = 0; o < 18; o++)
				{
					h1 = min(*src * n1, 0.2f);
					h2 = min(*src * n2, 0.2f);
					h3 = min(*src * n3, 0.2f);
					h4 = min(*src * n4, 0.2f);
					*dst2 = 0.5 * (h1 + h2 + h3 + h4);
					t1 += h1;
					t2 += h2;
					t3 += h3;
					t4 += h4;
					dst2 += out[0]*out[1];
					src += blocks[0]*blocks[1];
				}

				// contrast-insensitive features
				src = hist + (x+1)*blocks[0] + (y+1);

				for (int o = 0; o < 9; o++)
				{
					sum = *src + *(src + 9*blocks[0]*blocks[1]);
					h1 = min(sum * n1, 0.2f);
					h2 = min(sum * n2, 0.2f);
					h3 = min(sum * n3, 0.2f);
					h4 = min(sum * n4, 0.2f);
					*dst2 = 0.5 * (h1 + h2 + h3 + h4);
					dst2 += out[0]*out[1];
					src += blocks[0]*blocks[1];
				}

				// texture features
				*dst2 = 0.2357 * t1;
				dst2 += out[0]*out[1];
				*dst2 = 0.2357 * t2;
				dst2 += out[0]*out[1];
				*dst2 = 0.2357 * t3;
				dst2 += out[0]*out[1];
				*dst2 = 0.2357 * t4;

				// truncation feature
				dst2 += out[0]*out[1];
				*dst2 = 0;
			}
		}

		delete[] hist;
		delete[] norm;

		count = 0;
		std::vector<cv::Mat_<double> > tmp_mxfeat;
		cv::split(mxfeat, tmp_mxfeat);
		for (int ch = 0; ch < out[2]; ch++) {
			for (int col = 0; col < out[1]; col++) {
				for (int row = 0; row < out[0]; row++) {
					tmp_mxfeat[ch].at<double>(row, col) =  feat[count];
					// cout<<feat[count]<<endl;
					count++;
				}
			}
		}
		cv::merge(tmp_mxfeat, mxfeat);
        cv::copyMakeBorder(mxfeat, mxfeat,
            1, 1, 1, 1, cv::BORDER_REPLICATE);

		delete[] feat;
	}
	//
	// delete[] im;
	//
	return mxfeat;
}
