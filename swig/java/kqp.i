
%import "java/std_vector.i"

// --- Operators
%ignore operator=;
%rename(get) operator();

// Use enums
%include "enums.swg"

// Loading
%pragma(java) jniclasscode=%{
    
  

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
    
  public static String[] getLinkers() {
      return new String[] { "g++", "msvc" };
  }
  
  public static String[] getArchitectures() {
      return new String[] { System.getProperty("os.arch") };
  }
  
  public static boolean init(String kqplibname) {
      try {
          System.err.format("Trying to load %s%n", kqplibname);
          System.loadLibrary(kqplibname);
          System.err.println("Success.");
          return true;
      } catch (UnsatisfiedLinkError e) {
        System.err.println("Native code library failed to load. \n" + e);
      }
      
      return false;
  }
    
  public static void init() {
      String kqpname = System.getProperty("kqp.jni.path");
      if (kqpname != null)
        if (init(kqpname)) return;
      
      // Try simple things
      if (init("kqp_jni")) return;
      if (init("libkqp_jni")) return;
    
    // Try qualified names
    for(String arch: getArchitectures())
    for(String os: getOS())
    for(String linker: getLinkers()) 
        if (init(String.format("kqp-1.0.0-SNAPSHOT-%s-%s-%s-jni",arch,os,linker))) return;
        
    System.err.format("Could not find the KQP native library");
  }
  
  static {
    init();
  }

%}
