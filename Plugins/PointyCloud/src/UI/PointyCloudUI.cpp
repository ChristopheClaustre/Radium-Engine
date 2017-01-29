#include "PointyCloudUI.h"
#include "ui_PointyCloudUI.h"

#include <iostream>

PointyCloudUI::PointyCloudUI(float splatRadius, float influenceRadius, float beta, float Threshold,
                             PointyCloudPlugin::UPSAMPLING_METHOD upsampler,
                             PointyCloudPlugin::PROJECTION_METHOD projector,
                             bool cuda, bool octree, QWidget *parent) :
    QFrame(parent),
    ui(new Ui::PointyCloudUI)
{
//    ui->m_play->setProperty( "pressed", false );
//    ui->setupUi(this);
//    ui->m_play->style()->unpolish( ui->m_play );
//    ui->m_play->style()->polish( ui->m_play );
//    ui->m_play->update();
}

PointyCloudUI::~PointyCloudUI()
{
    delete ui;
}

