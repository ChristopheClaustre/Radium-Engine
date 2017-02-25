#include <Engine/Renderer/Mesh/Mesh.hpp>

#include <Core/Mesh/MeshUtils.hpp>
#include <Core/Mesh/HalfEdge.hpp>

namespace Ra {
    namespace Engine {

        // Dirty is initializes as false so that we do not create the vao while
        // we have no data to send to the gpu.
        Mesh::Mesh( const std::string& name, GLenum renderMode )
            : m_name( name )
            , m_vao( 0 )
            , m_renderMode(renderMode)
            , m_numElements (0)
            , m_isDirty( false )
        {
            CORE_ASSERT( m_renderMode == GL_LINES
                      || m_renderMode == GL_LINES_ADJACENCY
                      || m_renderMode == GL_TRIANGLES
                      || m_renderMode == GL_POINTS,
                         "Unsupported render mode" );
        }

        Mesh::~Mesh()
        {
            if (m_vao != 0)
            {
                GL_ASSERT( glDeleteVertexArrays( 1, &m_vao ) );

                for (auto& vbo : m_vbos)
                {
                    if (vbo != 0)
                    {
                        glDeleteBuffers(1, &vbo);
                    }
                }
            }
        }

        void Mesh::render()
        {
            if ( m_vao != 0 )
            {
                GL_ASSERT( glBindVertexArray( m_vao ) );
                GL_ASSERT( glDrawElements( m_renderMode, m_numElements, GL_UNSIGNED_INT, (void*)0 ) );
            }
        }

        void Mesh::loadGeometry(const Core::TriangleMesh& mesh)
        {
            m_mesh = mesh;
            m_numElements = mesh.m_triangles.size() * 3;
            for (uint i = 0; i < MAX_MESH; ++i)
            {
                m_dataDirty[i] = true;
            }
            m_isDirty = true;

        }

        void Mesh::loadGeometry(const Core::Vector3Array &vertices, const std::vector<uint> &indices)
        {
            // TODO : remove this function and force everyone to use triangle mesh.
            // Or not... because we have some line meshes as well...
            const uint nIdx = indices.size();
            m_numElements = nIdx;
            m_mesh.m_vertices = vertices;

            // Check that when loading a triangle mesh we actually have triangles.
            CORE_ASSERT( m_renderMode != GL_TRIANGLES || nIdx % 3 == 0, "There should be 3 indices per triangle " );
            CORE_ASSERT( m_renderMode != GL_LINES     || nIdx % 2 == 0, "There should be 2 indices per lines" );

            for ( uint i = 0; i < indices.size(); i = i + 3 )
            {
                // We store all indices in order. This means that for lines we have
                // (L00, L01, L10), (L11, L20, L21) etc. We fill the missing by wrapping around indices.
                m_mesh.m_triangles.push_back( { indices[i], indices[(i + 1)%nIdx], indices[(i + 2)%nIdx] } );
            }

            // Mark mesh as dirty.
            for (uint i = 0; i < MAX_MESH; ++i)
            {
                m_dataDirty[i] = true;
            }
            m_isDirty = true;
        }

        /// Use it to load an point-based geometry
        void Mesh::loadPointyGeometry(const Core::Vector3Array &vertices, const Core::Vector3Array &normals)
        {
            CORE_ASSERT( m_renderMode == GL_POINTS, "This function can be used only to render points");

            m_numElements = vertices.size();
            m_mesh.m_vertices = vertices;
            m_mesh.m_normals = normals;

            m_mesh.m_triangles.clear();
            for ( uint i = 0; i < vertices.size(); i += 3 )
            {
                m_mesh.m_triangles.push_back( { i, (i + 1)%m_numElements, (i + 2)%m_numElements } );
            }

            // Mark mesh as dirty.
            for (uint i = 0; i < MAX_MESH; ++i)
            {
                m_dataDirty[i] = true;
            }
            m_isDirty = true;
        }

        void Mesh::addData( const Vec3Data& type, const Core::Vector3Array& data )
        {
            m_v3Data[static_cast<uint>(type)] = data;
            m_dataDirty[MAX_MESH + static_cast<uint>(type)] = true;
            m_isDirty = true;
        }

        void Mesh::addData( const Vec4Data& type, const Core::Vector4Array& data )
        {
            m_v4Data[static_cast<uint>(type)] = data;
            m_dataDirty[MAX_MESH + MAX_VEC3 + static_cast<uint>(type)] = true;
            m_isDirty = true;
        }

        void Mesh::addData( const Vec1Data& type, const Core::Vector1Array& data )
        {
            m_v1Data[static_cast<uint>(type)] = data;
            m_dataDirty[MAX_MESH + MAX_VEC3 + MAX_VEC4 + static_cast<uint>(type)] = true;
            m_isDirty = true;
        }

        // Template parameter must be a Core::VectorNArray
        template< typename VecArray >
        void Mesh::sendGLData( const VecArray& arr, const uint vboIdx )
        {

#ifdef CORE_USE_DOUBLE
            GLenum type = GL_DOUBLE;
#else
            GLenum type = GL_FLOAT;
#endif
            constexpr GLuint size = VecArray::Vector::RowsAtCompileTime;
            constexpr GLboolean normalized  = GL_FALSE;
            constexpr GLint64 ptr = 0;

            // This vbo has not been created yet
            if ( m_vbos[vboIdx] == 0 && arr.size() > 0 )
            {
                GL_ASSERT( glGenBuffers( 1, &m_vbos[vboIdx] ) );
                GL_ASSERT( glBindBuffer( GL_ARRAY_BUFFER, m_vbos[vboIdx] ) );
                GL_ASSERT( glBufferData( GL_ARRAY_BUFFER, arr.size() * sizeof( typename VecArray::Vector ),
                                            arr.data(), GL_DYNAMIC_DRAW ) );

                // Use (vboIdx - 1) as attribute index because vbo 0 is actually ibo.
                GL_ASSERT( glVertexAttribPointer( vboIdx - 1, size, type, normalized,
                                                  sizeof( typename VecArray::Vector ), (GLvoid*)ptr ) );

                GL_ASSERT( glEnableVertexAttribArray( vboIdx - 1 ) );

            }

            if ( m_dataDirty[vboIdx] == true && m_vbos[vboIdx] != 0 && arr.size() > 0 )
            {
                GL_ASSERT( glBindBuffer( GL_ARRAY_BUFFER, m_vbos[vboIdx] ) );
                GL_ASSERT( glBufferData( GL_ARRAY_BUFFER, arr.size() * sizeof( typename VecArray::Vector ),
                arr.data(), GL_DYNAMIC_DRAW ) );
                m_dataDirty[vboIdx] = false;
            }
        }

        // "Dynamic Sampling and Rendering of APSS" students team
        // this function si needed for the moment because of the reality behind the typedef Vector1Array :
        // Vector1Array isn't an array of vector (of any dimension), it's just an array(std::vector) of scalar
        // and the templated sendGLData method use attributes only available in an Eigen's vector...
        // Maybe we should change the Vector1Array, to not have to hide this terrible reality ???
        void Mesh::sendGLData( const Core::Vector1Array& arr, const uint vboIdx )
        {
#ifdef CORE_USE_DOUBLE
            GLenum type = GL_DOUBLE;
#else
            GLenum type = GL_FLOAT;
#endif
            constexpr GLuint size = 1;
            constexpr GLboolean normalized  = GL_FALSE;
            constexpr GLint64 ptr = 0;

            // This vbo has not been created yet
            if ( m_vbos[vboIdx] == 0 && arr.size() > 0 )
            {
                GL_ASSERT( glGenBuffers( 1, &m_vbos[vboIdx] ) );
                GL_ASSERT( glBindBuffer( GL_ARRAY_BUFFER, m_vbos[vboIdx] ) );
                GL_ASSERT( glBufferData( GL_ARRAY_BUFFER, arr.size() * sizeof( Scalar ),
                                            arr.data(), GL_DYNAMIC_DRAW ) );

                // Use (vboIdx - 1) as attribute index because vbo 0 is actually ibo.
                GL_ASSERT( glVertexAttribPointer( vboIdx - 1, size, type, normalized,
                                                  sizeof( Scalar ), (GLvoid*)ptr ) );

                GL_ASSERT( glEnableVertexAttribArray( vboIdx - 1 ) );

            }

            if ( m_dataDirty[vboIdx] == true && m_vbos[vboIdx] != 0 && arr.size() > 0 )
            {
                GL_ASSERT( glBindBuffer( GL_ARRAY_BUFFER, m_vbos[vboIdx] ) );
                GL_ASSERT( glBufferData( GL_ARRAY_BUFFER, arr.size() * sizeof( Scalar ),
                arr.data(), GL_DYNAMIC_DRAW ) );
                m_dataDirty[vboIdx] = false;
            }
        }

        void Mesh::updateGL()
        {
            if ( m_isDirty )
            {
                // Check that our dirty bits are consistent.
                ON_DEBUG(bool dirtyTest = false; for (const auto& d : m_dataDirty) { dirtyTest = dirtyTest || d;});
                CORE_ASSERT( dirtyTest == m_isDirty, "Dirty flags inconsistency");

                CORE_ASSERT( !m_mesh.m_vertices.empty(), "Vertices is empty array.");
                CORE_ASSERT( !m_mesh.m_triangles.empty(), "Indices is empty array.");

                if ( m_vao == 0 )
                {
                    // Create VAO if it does not exist
                    GL_ASSERT( glGenVertexArrays( 1, &m_vao ) );
                }

                // Bind it
                GL_ASSERT( glBindVertexArray( m_vao ) );

                if (m_vbos[INDEX] == 0)
                {
                    GL_ASSERT( glGenBuffers( 1, &m_vbos[INDEX]) );
                    GL_ASSERT( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_vbos[INDEX]) );
                }
                if (m_dataDirty[INDEX])
                {
                    GL_ASSERT( glBufferData( GL_ELEMENT_ARRAY_BUFFER, m_mesh.m_triangles.size() * sizeof( Ra::Core::Triangle ),
                                             m_mesh.m_triangles.data(), GL_DYNAMIC_DRAW ) );
                    m_dataDirty[INDEX] = false;
                }

                // Geometry data
                sendGLData(m_mesh.m_vertices, VERTEX_POSITION);
                sendGLData(m_mesh.m_normals,  VERTEX_NORMAL);

                // Vec3 data
                sendGLData(m_v3Data[VERTEX_TANGENT],   MAX_MESH + VERTEX_TANGENT);
                sendGLData(m_v3Data[VERTEX_BITANGENT], MAX_MESH + VERTEX_BITANGENT);
                sendGLData(m_v3Data[VERTEX_TEXCOORD],  MAX_MESH + VERTEX_TEXCOORD);

                // Vec4 data
                sendGLData(m_v4Data[VERTEX_COLOR],      MAX_MESH + MAX_VEC3 + VERTEX_COLOR );
                sendGLData(m_v4Data[VERTEX_WEIGHTS],    MAX_MESH + MAX_VEC3 + VERTEX_WEIGHTS);
                sendGLData(m_v4Data[VERTEX_WEIGHT_IDX], MAX_MESH + MAX_VEC3 + VERTEX_WEIGHT_IDX);

                // Vec1 data
                sendGLData(m_v1Data[POINT_RADIUS], MAX_MESH + MAX_VEC3 + MAX_VEC4 + POINT_RADIUS);

                GL_ASSERT( glBindVertexArray( 0 ) );

                GL_CHECK_ERROR;
                m_isDirty = false;
            }
        }

    } // namespace Engine
} // namespace Ra
