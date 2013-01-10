a = [0.5309	0.6204	0.3487	0.3395	0.3395	0.3829	0.3620	0.3495	0.3478;
0.8462	0.8397	0.7699	0.7416	0.7416	0.7666	0.7579	0.7590	0.7601;
0.7921	0.7822	0.7426	0.6832	0.6832	0.6832	0.6733	0.6733	0.6634;
0.8774	0.9016	0.8273	0.8169	0.8169	0.8394	0.8048	0.8100	0.8100;
0.4170	0.4310	0.3891	0.3643	0.3643	0.3798	0.3597	0.3597	0.3597;
0.0201	0.0201	0.0201	0.0201	0.0201	0.0201	0.0201	0.0201	0.0201]

ranks = zeros(6,9);
for i=1:6
	[b,c] = sort(a(i,:));
	for r=9:-1:1
		sumc=0;
		countc=0;
		indices=[];
		for rr=r:9
			if a(i,c(r)) == a(i,c(rr))
				sumc = sumc + rr;
				countc = countc+1;
				indices=[indices; c(rr)];
			end
		end
		c2(indices) = sumc/countc;
	end
	
	ranks(i,:) = c2;
end
averages=zeros(1,9);
for i=1:9
	averages(i) = mean(ranks(:,i));
end
ranks
averages
