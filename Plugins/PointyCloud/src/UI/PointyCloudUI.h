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
                           PointyCloudPlugin::UPSAMPLING_METHOD upsampler,
                           PointyCloudPlugin::PROJECTION_METHOD projector,
                           bool cuda, bool octree, QWidget *parent = 0);
    ~PointyCloudUI();

signals:
    void setSplatRadius(float);
    void setInfluenceRadius(float);
    void setBeta(float);
    void setThreshold(float);
    void setUpsamplingMethod(PointyCloudPlugin::UPSAMPLING_METHOD);
    void setProjectionMethod(PointyCloudPlugin::PROJECTION_METHOD);
    void setOptimizationByOctree(bool);
    void setOptimizationByCUDA(bool);

private slots:


private:
    Ui::PointyCloudUI *ui;
};

#endif // POINTYCLOUDUI_H
