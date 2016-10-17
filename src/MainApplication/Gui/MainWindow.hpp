#ifndef RADIUMENGINE_MAINWINDOW_HPP
#define RADIUMENGINE_MAINWINDOW_HPP

#include <QMainWindow>
#include "ui_MainWindow.h"

#include <qdebug.h>
#include <QEvent>

#include <GuiBase/TreeModel/EntityTreeModel.hpp>
#include <GuiBase/SelectionManager/SelectionManager.hpp>
#include <MainApplication/TimerData/FrameTimerData.hpp>

namespace Ra
{
    namespace Engine
    {
        class Entity;
    }
}

namespace Ra
{
    namespace Gui
    {
        class EntityTreeModel;
        class Viewer;
        class MaterialEditor;
    }
}

namespace Ra
{
    namespace Plugins
    {
        class RadiumPluginInterface;
    }
}

namespace Ra
{
    namespace Gui
    {
        /// This class manages most of the GUI of the application :
        /// top menu, side toolbar and side dock.
        class MainWindow : public QMainWindow, private Ui::MainWindow
        {
            Q_OBJECT

        public:
            /// Constructor and destructor.
            MainWindow(uint fps, std::string pluginsPath, uint numFrames, std::string fileToLoad = "", QWidget* parent = nullptr );
            virtual ~MainWindow();

            /// Access the viewer, i.e. the rendering widget.
            Viewer* getViewer();

            /// Access the selection manager.
            GuiBase::SelectionManager* getSelectionManager();

            /// Update the ui from the plugins loaded.
            void updateUi( Plugins::RadiumPluginInterface* plugin );

            /// Update the UI ( most importantly gizmos ) to the modifications of the engine/
            void onFrameComplete();

            void loadFile(std::string file);
            bool loadPlugins();
            void setupScene();
            void addBasicShaders();

        public slots:
            /// Callback to rebuild the item model when the engine objects change.
            void onItemAdded( const Engine::ItemEntry& ent );

            void onItemRemoved( const Engine::ItemEntry& ent );

            // Frame timers ui slots
            void updateFrameStats( const std::vector<FrameTimerData>& stats );

            // Selection tools
            void onSelectionChanged( const QItemSelection& selected, const QItemSelection& deselected );

            // Gizmo buttons slots
            void gizmoShowNone();
            void gizmoShowTranslate();
            void gizmoShowRotate();
            //void gizmoShowScale();

            /// Slot for the "visible" button
            void toggleVisisbleRO();

            /// Slot for the "edit" button.
            void editRO();

            /// Cleanup resources.
            void cleanup();

        signals:
            /// Emitted when the closed button has been hit.
            void closed();

            /// Emitted when the user changes the timer box ("Frame average over count")
            void framescountForStatsChanged( int count );

            /// Emitted when a new item is selected. An invalid entry is sent when no item is selected.
            void selectedItem( const Engine::ItemEntry& entry );

        private:
            /// Connect qt signals and slots. Called once by the constructor.
            void createConnections();

            virtual void closeEvent( QCloseEvent* event ) override;

        private slots:
            /// Slot for the "load file" menu.
            void openMesh();

            /// Slot for the "material editor"
            void openMaterialEditor();

            /// Slot for the user changing the current shader
            void changeRenderObjectShader(const QString& shaderName);

            /// Slot for the picking results from the viewer.
            void handlePicking( int ROIndex );

            /// Slot to accept a new renderer
            void onRendererReady();

            void setRealFrameRate(bool on) { m_useRealFramerate = on; }

            /// Main loop
            void radiumFrame();

            void framesCountForStatsChanged(uint value)
            {
                m_frameCountBeforeUpdate = value;
            }

        public:
            std::unique_ptr<Engine::RadiumEngine> m_engine;
            std::unique_ptr<Core::TaskQueue> m_taskQueue;

            uint m_targetFPS;

        private:
            /// Stores the internal model of engine objects for selection.
            GuiBase::ItemModel* m_itemModel;

            /// Stores and manage the current selection.
            GuiBase::SelectionManager* m_selectionManager;

            /// Widget to allow material edition.
            MaterialEditor* m_materialEditor;

            /// Timer
            QTimer* m_timer;

            Ra::Core::Timer::TimePoint m_lastFrameStart;

            uint m_frameCounter;
            uint m_frameCountBeforeUpdate;
            uint m_numFrames;

            std::string m_pluginsPath;

            std::vector<FrameTimerData> m_timerData;

            bool m_useRealFramerate;
        };
    } // namespace Gui
} // namespace Ra

#endif // RADIUMENGINE_MAINWINDOW_HPP
