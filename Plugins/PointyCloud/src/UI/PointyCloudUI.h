#ifndef POINTYCLOUDUI_H
#define POINTYCLOUDUI_H

#include <PointyCloudPlugin.hpp>
#include <QFrame>

namespace Ui {
    class PointyCloudUI;
}

class PointyCloudUI : public QFrame
{
    Q_OBJECT

public:
    explicit PointyCloudUI(float splatRadius, float influenceRadius, float beta, float Threshold,
    explicit PointyCloudUI(float splatRadius, float influenceRadius, float beta, float Threshold, int M,
                           PointyCloudPlugin::UPSAMPLING_METHOD upsampler,
                           PointyCloudPlugin::PROJECTION_METHOD projector,
                           bool cuda, bool octree, QWidget *parent = 0);
    ~PointyCloudUI();

signals:
    void setSplatRadius(float);
    void setInfluenceRadius(float);
    void setBeta(float);
    void setThreshold(float);
    void setThreshold(int);
    void setM(int);
    void setUpsamplingMethod(PointyCloudPlugin::UPSAMPLING_METHOD);
    void setProjectionMethod(PointyCloudPlugin::PROJECTION_METHOD);
    void setOptimizationByOctree(bool);
    void setOptimizationByCUDA(bool);

private slots:

    void on_m_splatRadius_editingFinished();
    void on_m_influenceRadius_editingFinished();
    void on_m_beta_editingFinished();
    void on_m_threshold_editingFinished();
    void on_m_upsamplingMethodes_currentIndexChanged(int index);
    void on_m_projectionMethodes_currentIndexChanged(int index);
    void on_m_upsamplingMethod_currentIndexChanged(int index);
    void on_m_projectionMethod_currentIndexChanged(int index);
    void on_m_octree_clicked(bool checked);
    void on_m_cuda_clicked(bool checked);

private:
    bool isClampValideDValue(double value,double min,double max);
    bool isClampValideIValue(int value,int min,int max);
    double clampDValue(double value,double min,double max);
    int clampIValue(int value,int min,int max);

private:
    Ui::PointyCloudUI *ui;
};

#endif // POINTYCLOUDUI_H
