
%import "java/std_vector.i"

// --- Operators
%ignore operator=;
%rename(get) operator();

// Pointers to primitive types
%include "cpointer.i"

// Type maps
%include "typemaps.i" 


// Use enums
%include "enums.swg"

// Loading
%pragma(java) jniclassimports=%{
    import java.io.BufferedReader;
    import java.io.File;
    import java.io.IOException;
    import java.io.InputStreamReader;
    import java.net.URL;
    import java.util.ArrayList;
    import java.util.Enumeration;
    import java.util.Map;
    import java.util.Properties;
%}

%pragma(java) jniclasscode=%{
    

    private static final String KQP_PROPERTIES = "META-INF/services/net.bpiwowar.kqp";

    // ---  Generate the Architecture-OS string
    public static String[] getOS()
    {
        String name = System.getProperty( "os.name" ).toLowerCase();
        if ( name.startsWith( "windows" ) )
            return new String[] { "Windows" };
        if ( name.startsWith( "linux" ) )
            return new String[] { "Linux" };
        if ( name.equals( "mac os x" ) )
            return new String[] { "MacOSX" };

        return new String[] { "Windows", "MacOSX", "Linux", "SunOS" };
    }


    public static String[] getArchitectures() {
        return new String[] { System.getProperty("os.arch") };
    }


    static ArrayList<String> SYSTEM_IDS = new ArrayList<String>();
    static {
        for(String arch: getArchitectures())
            for(String os: getOS())
                SYSTEM_IDS.add(String.format("%s-%s",arch,os));
    }

    static ArrayList<String> JNIEXT = new ArrayList<String>();
    static {
        JNIEXT.add("jnilib");
        JNIEXT.add("so");
    }

    /** Try to init KQP with a given name */
    public static boolean init(String kqplibname) {
        try {
            System.err.format("[KQP] Trying to load JNI library %s%n", kqplibname);
            if (new File(kqplibname).isFile())
                System.load(kqplibname);
            else
                System.loadLibrary(kqplibname);
            System.err.println("[KQP] Success.");
            return true;
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load. \n" + e);
        }

        return false;
    }

    private static boolean init(Properties properties) {
        String libraryPath = properties.get("library.path").toString();
        if (libraryPath != null) {
            for(String systemId: SYSTEM_IDS)
            for(String jniext: JNIEXT) {
                libraryPath = libraryPath.replaceAll("@systemId@", systemId);
                libraryPath = libraryPath.replaceAll("@jniext@", jniext);
                if (init(libraryPath)) return true;
            }
        }
        else System.out.println("No library path");
        return false;
    }


    private static void loadKQP() {
        // Try with system property
        String kqpname = System.getProperty("kqp.jni.path");
        if (kqpname != null)
            if (init(kqpname)) return;

        // Try with stored properties
        try {
            final Enumeration<URL> e = kqpJNI.class.getClassLoader().getResources(KQP_PROPERTIES);
            while (e.hasMoreElements()) {
                URL url = e.nextElement();
                BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()));
                String basepath = null;
                if (url.getProtocol() == "jar") {
                    url = new URL(url.getPath());
                    basepath = new File((url.getPath().split("!"))[0]).getParentFile().toString();
                }
                Properties properties = new Properties();
                properties.load(in);
                for(Map.Entry<Object, Object> p: properties.entrySet()) {
                    p.setValue(p.getValue().toString().replaceAll("@basepath@", basepath));
                }
                if (init(properties)) return;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


        // Try simple things
        if (init("kqp_jni")) return;
        if (init("libkqp_jni")) return;

        // Try qualified names
        for(String systemId: SYSTEM_IDS)
            if (init(String.format("kqp-1.0.0-SNAPSHOT-%s-jni",systemId))) return;


        System.err.format("Could not find the KQP native library");
    }

    static {
      loadKQP();
    }


%}


// Map iterator, source
// http://stackoverflow.com/questions/9465856/no-iterator-for-java-when-using-swig-with-cs-stdmap


%typemap(javainterfaces) kqp::MapIterator<Index,Index> "java.util.Iterator<LongLongPair>"
%typemap(javacode) kqp::MapIterator<Index,Index> %{
  public void remove() throws UnsupportedOperationException {
    throw new UnsupportedOperationException();
  }

  public LongLongPair next() throws java.util.NoSuchElementException {
    if (!hasNext()) {
      throw new java.util.NoSuchElementException();
    }

    return nextImpl();
  }
%}

%javamethodmodifiers LongLongMapIterator::nextImpl "private";
%inline %{
    namespace kqp {
        template<typename Key, typename Value>
  struct MapIterator {
    typedef std::map<Index, Index> map_t;
    
    MapIterator(const map_t& m) : it(m.begin()), map(m) {}
    bool hasNext() const {
      return it != map.end();
    }

    const std::pair<Index, Index>  nextImpl() {
      return *it++;
    }
  private:
    map_t::const_iterator it;
    const map_t& map;    
  };
  }
%}

%template(LongLongMapIterator) kqp::MapIterator<Index,Index>;
%template(LongLongPair) std::pair<Index,Index>;

// Say that the object implement the interface
%typemap(javainterfaces) std::map<Index,Index> "Iterable<LongLongPair>"

// Extend std::map
%newobject std::map<Index,Index>::iterator() const;
%extend std::map<Index,Index> {
  kqp::MapIterator<Index,Index> *iterator() const {
    return new kqp::MapIterator<Index,Index>(*$self);
  }
}


