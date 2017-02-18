#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDUI_H
#define POINTYCLOUDPLUGIN_POINTYCLOUDUI_H

#include <PointyCloudPlugin.hpp>
#include <QFrame>

namespace Ui {
    class PointyCloudUI;
}

class PointyCloudUI : public QFrame
{
    Q_OBJECT

public:
    explicit PointyCloudUI(Scalar splatRadius, Scalar influenceRadius, Scalar beta, int Threshold, int M,
                           PointyCloudPlugin::UPSAMPLING_METHOD upsampler,
                           PointyCloudPlugin::PROJECTION_METHOD projector,
                           bool cuda, bool octree, QWidget *parent = 0);
    ~PointyCloudUI();

signals:
    void setSplatRadius(Scalar);
    void setInfluenceRadius(Scalar);
    void setBeta(Scalar);
    void setThreshold(int);
    void setM(int);
    void setUpsamplingMethod(PointyCloudPlugin::UPSAMPLING_METHOD);
    void setProjectionMethod(PointyCloudPlugin::PROJECTION_METHOD);
    void setOptimizationByOctree(bool);
    void setOptimizationByCUDA(bool);

private slots:

    void on_m_splatRadius_valueChanged(double value);
    void on_m_influenceRadius_valueChanged(double value);
    void on_m_beta_valueChanged(double value);
    void on_m_threshold_valueChanged(int value);
    void on_m_M_valueChanged(int value);
    void on_m_upsamplingMethod_currentIndexChanged(int index);
    void on_m_projectionMethod_currentIndexChanged(int index);
    void on_m_octree_clicked(bool checked);
    void on_m_cuda_clicked(bool checked);

private:
    Ui::PointyCloudUI *ui;
};

#endif // POINTYCLOUDUI_H
