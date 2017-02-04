#include "PointyCloudUI.h"
#include "PointyCloudPlugin.hpp"
#include "ui_PointyCloudUI.h"

#include <iostream>

PointyCloudUI::PointyCloudUI(float splatRadius, float influenceRadius, float beta, float Threshold,
                             PointyCloudPlugin::UPSAMPLING_METHOD upsampler,
                             PointyCloudPlugin::PROJECTION_METHOD projector,
                             bool cuda, bool octree, QWidget *parent) :
    QFrame(parent),
    ui(new Ui::PointyCloudUI)
{
    ui->setupUi(this);
    ui->m_splatRadius->setValue(splatRadius);
    ui->m_influenceRadius->setValue(influenceRadius);
    ui->m_beta->setValue(beta);
    ui->m_threshold->setValue(Threshold);

    //Load list upsampler methods
    for(std::string method : PointyCloudPlugin::PointyCloudPluginC::UPSAMPLING_METHOD_STR){
        ui->m_upsamplingMethodes->addItem(QString::fromStdString(method));
    }
    ui->m_upsamplingMethodes->setCurrentIndex(upsampler);
    //Load list projection methods
    for(std::string method : PointyCloudPlugin::PointyCloudPluginC::PROJECTION_METHOD_STR){
        ui->m_projectionMethodes->addItem(QString::fromStdString(method));
    }
    ui->m_projectionMethodes->setCurrentIndex(projector);
    ui->m_cuda->setChecked(cuda);
    ui->m_octree->setChecked(octree);
}

PointyCloudUI::~PointyCloudUI()
{
    delete ui;
}

//TODO (xavier): modifier valeurs de clamp par varibles static de system et modifier la technique de clamping
void PointyCloudUI::on_m_splatRadius_editingFinished()
{
    double splatRaduius = ui->m_splatRadius->value();
    //Clamp value
    float min = 0.0f;
    float max = 30.0f; //NOTE(chris): les cartes graphiques ont une limite haute pour ce paramètre ;) ;)
    if(!isClampValideDValue(splatRaduius,min,max)){
        //NOTE(chris): cette fonction ne rapelle pas editingFinished après donc ca le signal n'es jamais envoyé
        ui->m_splatRadius->setValue(clampDValue(splatRaduius,min,max));
    }
    else
        emit setSplatRadius(splatRaduius);
}

void PointyCloudUI::on_m_influenceRadius_editingFinished()
{
    double influenceRadius = ui->m_influenceRadius->value();
    //Clamp value
    float min = 0.0f;
    float max = 10.0f;
    if(!isClampValideDValue(influenceRadius,min,max))
        //NOTE(chris): cette fonction ne rapelle pas editingFinished après donc ca le signal n'es jamais envoyé
        ui->m_influenceRadius->setValue(clampDValue(influenceRadius,min,max));
    else
        emit setInfluenceRadius(influenceRadius);
}

void PointyCloudUI::on_m_beta_editingFinished()
{
    double beta = ui->m_beta->value();
    //Clamp value
    float min = 1.0f;
    float max = 2.0f;
    if(!isClampValideDValue(beta,min,max))
        //NOTE(chris): cette fonction ne rapelle pas editingFinished après donc ca le signal n'es jamais envoyé
        ui->m_beta->setValue(clampDValue(beta,min,max));
    else
        emit setBeta(beta);
}


void PointyCloudUI::on_m_threshold_editingFinished()
{
    int threshold = ui->m_threshold->value();
    //Clamp value
    int min = 0;
    int max = 5;
    if(!isClampValideIValue(threshold,min,max))
        //NOTE(chris): cette fonction ne rapelle pas editingFinished après donc ca le signal n'es jamais envoyé
        ui->m_threshold->setValue(clampIValue(threshold,min,max));
    else
        emit setThreshold(threshold);
}

void PointyCloudUI::on_m_upsamplingMethodes_currentIndexChanged(int index)
{
    emit setUpsamplingMethod(PointyCloudPlugin::UPSAMPLING_METHOD(index));
}

void PointyCloudUI::on_m_projectionMethodes_currentIndexChanged(int index)
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

//Fonctions utilitaires (à deplacer ?)
//NOTE(chris): nom de fonction un peut pourri non ??
bool PointyCloudUI::isClampValideDValue(double value,double min,double max)
{
    //NOTE(chris): si value == min ou == max ça retourne false, est ce vraiment nécessaire ?
    return (value > min && value < max);
}

bool PointyCloudUI::isClampValideIValue(int value,int min,int max)
{
    return (value > min && value < max);
}

double PointyCloudUI::clampDValue(double value,double min,double max)
{
    if(value < min)
        return min;
    else if(value > max)
        return max;
    return value;
}

int PointyCloudUI::clampIValue(int value,int min,int max)
{
    if(value < min)
        return min;
    else if(value > max)
        return max;
    return value;
}
