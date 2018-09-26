/* jshint esversion: 6 */

module.exports = (grunt) => {
    grunt.loadNpmTasks("grunt-contrib-jshint");
    grunt.initConfig({
        jshint: {
            all: ["Gruntfile.js"]
        }
    });
};
