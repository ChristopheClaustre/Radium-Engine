#include "PointyCloudUI.h"
#include "PointyCloudPlugin.hpp"
#include "ui_PointyCloudUI.h"

#include <iostream>

PointyCloudUI::PointyCloudUI(Scalar splatRadius, Scalar influenceRadius, int Threshold, int M,
                             PointyCloudPlugin::UPSAMPLING_METHOD upsampler,
                             PointyCloudPlugin::PROJECTION_METHOD projector,
                             bool cuda, bool octree, bool APSS, bool renderer, QWidget *parent) :
    QFrame(parent),
    ui(new Ui::PointyCloudUI)
{
    ui->setupUi(this);
    ui->m_splatRadius->setValue(splatRadius);
    ui->m_splatRadius->setRange(PointyCloudPlugin::PointyCloudPluginC::splatRadiusInit.min, PointyCloudPlugin::PointyCloudPluginC::splatRadiusInit.max);
    ui->m_splatRadius->setSingleStep(PointyCloudPlugin::PointyCloudPluginC::splatRadiusInit.step);

    ui->m_influenceRadius->setValue(influenceRadius);
    ui->m_influenceRadius->setRange(PointyCloudPlugin::PointyCloudPluginC::influenceInit.min, PointyCloudPlugin::PointyCloudPluginC::influenceInit.max);
    ui->m_influenceRadius->setSingleStep(PointyCloudPlugin::PointyCloudPluginC::influenceInit.step);

    ui->m_threshold->setValue(Threshold);
    ui->m_threshold->setRange(PointyCloudPlugin::PointyCloudPluginC::thresholdInit.min, PointyCloudPlugin::PointyCloudPluginC::thresholdInit.max);
    ui->m_threshold->setSingleStep(PointyCloudPlugin::PointyCloudPluginC::thresholdInit.step);

    ui->m_M->setValue(M);
    ui->m_M->setRange(PointyCloudPlugin::PointyCloudPluginC::mInit.min, PointyCloudPlugin::PointyCloudPluginC::mInit.max);
    ui->m_M->setSingleStep(PointyCloudPlugin::PointyCloudPluginC::mInit.step);

    //Load list upsampler methods
    for(std::string method : PointyCloudPlugin::PointyCloudPluginC::UPSAMPLING_METHOD_STR){
        ui->m_upsamplingMethod->addItem(QString::fromStdString(method));
    }
    ui->m_upsamplingMethod->setCurrentIndex(upsampler);

    //Load list projection methods
    for(std::string method : PointyCloudPlugin::PointyCloudPluginC::PROJECTION_METHOD_STR){
        ui->m_projectionMethod->addItem(QString::fromStdString(method));
    }
    ui->m_projectionMethod->setCurrentIndex(projector);

    ui->m_cuda->setChecked(cuda);
    ui->m_octree->setChecked(octree);

    ui->m_APSS->setChecked(APSS);
    ui->m_renderer->setChecked(renderer);
}

PointyCloudUI::~PointyCloudUI()
{
    delete ui;
}

void PointyCloudUI::on_m_splatRadius_valueChanged(double value)
{
    emit setSplatRadius(ui->m_splatRadius->value());
}

void PointyCloudUI::on_m_influenceRadius_valueChanged(double value)
{
    emit setInfluenceRadius(value);
}

void PointyCloudUI::on_m_threshold_valueChanged(int value)
{
    emit setThreshold(value);
}

void PointyCloudUI::on_m_M_valueChanged(int value)
{
    emit setM(value);
}

void PointyCloudUI::on_m_upsamplingMethod_currentIndexChanged(int index)
{
        ui->label_8->setVisible(PointyCloudPlugin::UPSAMPLING_METHOD(index) == PointyCloudPlugin::FIXED_METHOD);
        ui->m_M->setVisible(PointyCloudPlugin::UPSAMPLING_METHOD(index) == PointyCloudPlugin::FIXED_METHOD);

        ui->label_4->setVisible(PointyCloudPlugin::UPSAMPLING_METHOD(index) == PointyCloudPlugin::SIMPLE_METHOD);
        ui->m_threshold->setVisible(PointyCloudPlugin::UPSAMPLING_METHOD(index) == PointyCloudPlugin::SIMPLE_METHOD);

        emit setUpsamplingMethod(PointyCloudPlugin::UPSAMPLING_METHOD(index));
}

void PointyCloudUI::on_m_projectionMethod_currentIndexChanged(int index)
{
    emit setProjectionMethod(PointyCloudPlugin::PROJECTION_METHOD(index));
}

void PointyCloudUI::on_m_octree_clicked(bool checked)
{
    emit setOptimizationByOctree(checked);
}

void PointyCloudUI::on_m_cuda_clicked(bool checked)
{
    emit setOptimizationByCUDA(checked);
}

void PointyCloudUI::on_m_APSS_clicked(bool checked)
{
    emit setAPSS(checked);
}

void PointyCloudUI::on_m_renderer_clicked(bool checked)
{
    emit setRenderer(checked);
}
