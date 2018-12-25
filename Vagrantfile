# -*- mode: ruby -*-
# vi: set ft=ruby :

unless Vagrant.has_plugin?("vagrant-docker-compose")
  abort "docker_compose plugin is missing, please run 'vagrant plugin install vagrant-docker-compose'"
end

unless Vagrant.has_plugin?("vagrant-vbguest")
  abort "vbguest plugin is missing, please run 'vagrant plugin install vagrant-vbguest'"
end

Vagrant.configure(2) do |config|
  config.vm.box = "ubuntu/trusty64"

  config.vm.define "dev"
  config.vm.hostname = "dev"

  # VirtualBox specific configuration
  config.vm.provider "virtualbox" do |vb|
    vb.memory = 2048
    vb.cpus = 4
  end

  # A few small Vagrant/Ubuntu related fixes, to make usage easier
  config.ssh.insert_key = false
  config.vm.provision "fix-no-tty", type: "shell" do |s|
    s.privileged = false
    s.inline = "sudo sed -i '/tty/!s/mesg n/tty -s \\&\\& mesg n/' /root/.profile"
  end

  # Provision Docker
  config.vm.provision :docker
  
  # Provision Docker Compose and then run the app containers
  config.vm.provision :docker_compose,
    yml: "/vagrant/docker-compose.yml",
    rebuild: true,
    run: "always"
end
