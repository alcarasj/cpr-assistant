module.exports = (grunt) => {
    require("load-grunt-tasks")(grunt);
    grunt.initConfig({
        eslint: {
            options: {
                reset: true
            },
            target: ["./*.js", "./screens/*.js"]
        }
    });
    grunt.registerTask("default", ["eslint"]);
};